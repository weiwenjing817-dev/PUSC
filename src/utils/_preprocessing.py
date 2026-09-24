from collections.abc import Sequence
from typing import Any, Literal
import time

import anndata as ad
import numpy as np
import sklearn.preprocessing as preprocessing

from src.utils._logging import logger
from src.utils._types import ArrayLike
from src.utils._utils import _to_list
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

__all__ = ["encode_onehot", "annotate_compounds", "get_molecular_fingerprints"]


def _normalize_pubchem_cid(cid) -> str:
    """Normalize numeric PubChem identifiers read from dataframe columns."""
    try:
        return str(int(float(cid)))
    except (TypeError, ValueError):
        return str(cid)


def get_smiles_pubchemlite(cid, timeout: float = 15.0):
    url = f"https://pubchemlite.lcsb.uni.lu/e/compound/{cid}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        label = soup.find(string="SMILES")
        return label.find_next().text.strip() if label is not None else None
    except (requests.RequestException, AttributeError):
        return None
    

def _get_smiles_via_ncbi(cid, timeout: float):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pccompound',
        'id': cid,
        'rettype': 'docsum',
        'retmode': 'xml'
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    for item in root.findall('.//Item'):
        if item.get('Name') in ('IsomericSmiles', 'CanonicalSMILES'):
            return item.text
    return None


def _get_smiles_via_pubchem_pug(cid, timeout: float):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
        "property/IsomericSMILES,CanonicalSMILES/JSON"
    )
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    properties = r.json().get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return None
    return properties[0].get("SMILES") or properties[0].get("ConnectivitySMILES")


def get_smiles_via_http(cid, retries: int = 3, timeout: float = 15.0):
    """Resolve a PubChem CID without failing preprocessing on transient HTTP errors."""
    cid = _normalize_pubchem_cid(cid)
    resolvers = (_get_smiles_via_ncbi, _get_smiles_via_pubchem_pug)
    for attempt in range(retries):
        for resolver in resolvers:
            try:
                smiles = resolver(cid, timeout)
                if smiles:
                    return smiles
            except (requests.RequestException, ET.ParseError, ValueError, KeyError):
                continue
        if attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))
    return get_smiles_pubchemlite(cid, timeout=timeout)


def annotate_compounds(
    adata: ad.AnnData,
    compound_keys: str | Sequence[str],
    control_category: str = "control",
    query_id_type: Literal["name", "cid"] = "name",
    obs_key_prefixes: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Annotates compounds in ``adata`` using pertpy and PubChem.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    compound_keys
        Key(s) in :attr:`~anndata.AnnData.obs` containing the compound identifiers.
    control_category
        Category to exclude from the annotation.
    query_id_type
        Type of the compound identifiers. Either ``'name'`` or ``'cid'``.
    obs_key_prefixes
        Prefix for the keys in :attr:`~anndata.AnnData.obs` to store the annotations. If :obj:`None`,
        uses ``compound_keys`` as prefixes.
    copy
        Return a copy of ``adata`` instead of updating it in place.

    Returns
    -------
        If ``copy`` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the compound
        annotations stored in :attr:`~anndata.AnnData.obs`. Otherwise, updates ``adata`` in place.

        Sets the following fields for each value in ``compound_keys``:

        - ``.obs[f"{obs_key_prefix}_pubchem_name"]``: Name of the compound.
        - ``.obs[f"{obs_key_prefix}_pubchem_ID"]``: PubChem CID of the compound.
        - ``.obs[f"{obs_key_prefix}_smiles"]``: SMILES representation of the compound.
    """
    try:
        import pertpy as pt
    except ImportError as e:
        raise ImportError(
            "pertpy is not installed. To annotate compounds, please install it via `pip install pertpy`."
        ) from e
    
    adata = adata.copy() if copy else adata

    compound_keys = _to_list(compound_keys)
    obs_key_prefixes = _to_list(obs_key_prefixes) if obs_key_prefixes is not None else compound_keys

    if len(compound_keys) != len(obs_key_prefixes):
        raise ValueError("The number of `compound_keys` must match the number of values in `obs_key_prefixes`.")

    # Annotate compounds in each query column
    not_found = set()
    c_meta = pt.metadata.Compound()
    for query_key, prefix in zip(compound_keys, obs_key_prefixes, strict=False):
        adata.obs[query_key] = adata.obs[query_key].replace(control_category, np.nan)
        c_meta.annotate_compounds(
            adata,
            query_id=query_key,
            query_id_type=query_id_type,
            verbosity=0,
            copy=False,
        )
        
        # Pertpy usually supplies SMILES. Only query PubChem for genuinely missing
        # values so a transient external-service failure cannot discard valid data.
        missing_smiles = adata.obs["smiles"].isna() | adata.obs["smiles"].astype(str).str.strip().eq("")
        missing_pubchem_ids = (
            adata.obs.loc[missing_smiles, "pubchem_ID"].dropna().unique().tolist()
        )

        for cid in missing_pubchem_ids:
            smiles = get_smiles_via_http(cid)
            if smiles:
                cid_mask = adata.obs["pubchem_ID"] == cid
                adata.obs.loc[cid_mask & missing_smiles, "smiles"] = smiles
            else:
                not_found.update(
                    adata.obs.loc[adata.obs["pubchem_ID"] == cid, query_key].dropna().astype(str)
                )
        
        adata.obs[[query_key, "pubchem_name"]] = adata.obs[[query_key, "pubchem_name"]].fillna(control_category)

        # Drop columns with new annotations
        adata.obs.rename(
            columns={
                "pubchem_name": f"{prefix}_pubchem_name",
                "pubchem_ID": f"{prefix}_pubchem_ID",
                "smiles": f"{prefix}_smiles",
            },
            inplace=True,
        )

    if not_found:
        logger.warning(
            "Could not resolve SMILES for the following compounds; their molecular "
            f"fingerprints will be omitted: {', '.join(sorted(not_found))}"
        )

    return adata if copy else None


def _get_fingerprint(smiles: str, radius: int = 4, n_bits: int = 1024) -> ArrayLike | None:
    """Computes Morgan fingerprints for a given SMILES string."""
    try:
        import rdkit.Chem.rdFingerprintGenerator as rfg
        from rdkit import Chem
    except ImportError:
        raise ImportError(
            "rdkit is not installed. To compute fingerprints, please install it via `pip install rdkit`."
        ) from None

    mmol = Chem.MolFromSmiles(str(smiles), sanitize=True)

    # Check if molecule is valid, MolFromSmiles returns None if error occurs
    if mmol is None:
        return None

    mfpgen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(mfpgen.GetFingerprint(mmol))


def get_molecular_fingerprints(
    adata,
    compound_keys: str | list[str],
    smiles_keys: str | None = None,
    control_value: str = "control",
    uns_key_added: str = "fingerprints",
    radius: int = 4,
    n_bits: int = 1024,
    copy: bool = False,
) -> None | ad.AnnData:
    """Computes Morgan fingerprints for compounds in ``adata`` and stores them in :attr:`~anndata.AnnData.uns`.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    compound_keys
        Key(s) in :attr:`~anndata.AnnData.obs` containing the compound identifiers.
    control_value
        Skip `control_value` (and :obj:`None` values).
    smiles_keys
        Key(s) in :attr:`~anndata.AnnData.obs` containing the SMILES strings. If :obj:`None`, uses
        ``f"{compound_key}_smiles"``.
    uns_key_added
        Key in :attr:`~anndata.AnnData.uns` to store the fingerprints.
    radius
        Radius of the Morgan fingerprints.
    n_bits
        Number of bits in the fingerprint.
    copy
        Return a copy of ``adata`` instead of updating it in place.

    Returns
    -------
        Updates :attr:`~anndata.AnnData.uns` with the computed fingerprints.

        Sets the following fields:

        - ``.uns[uns_key_added]``: Dictionary containing the fingerprints for each compound.
    """
    adata = adata.copy() if copy else adata

    compound_keys = _to_list(compound_keys)

    if smiles_keys is None:
        smiles_keys = [f"{key}_smiles" for key in compound_keys]

    smiles_keys = _to_list(smiles_keys)

    # Get dict with SMILES for each compound
    smiles_dict = {}
    for compound_key, smiles_key in zip(compound_keys, smiles_keys, strict=False):  # type: ignore[arg-type]
        if compound_key not in adata.obs:
            raise KeyError(f"Key {compound_key} not found in `adata.obs`.")

        if smiles_key not in adata.obs:
            raise KeyError(f"Key {smiles_key} not found in `adata.obs`.")
        smiles_dict.update(adata.obs.set_index(compound_key)[smiles_key].to_dict())

    # Compute fingerprints for each compound
    valid_fingerprints = {}
    not_found = []
    for comp, smiles in smiles_dict.items():
        if not isinstance(comp, str) or comp == control_value:
            continue
        comp_fp = _get_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if comp_fp is not None:
            valid_fingerprints[comp] = comp_fp
        else:
            not_found.append(str(comp))

    if not_found:
        logger.warning(f"Could not compute fingerprints for the following compounds: {', '.join(not_found)}")

    adata.uns[uns_key_added] = valid_fingerprints

    if copy:
        return adata


def encode_onehot(
    adata: ad.AnnData,
    covariate_keys: str | Sequence[str],
    uns_key_added: str,
    exclude_values: str | Sequence[Any] = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Encodes covariates :attr:`~anndata.AnnData.obs` as one-hot vectors and stores them in :attr:`~anndata.AnnData.uns`.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    covariate_keys
        Key(s) in :attr:`~anndata.AnnData.obs` containing the covariate(s) to encode.
    uns_key_added
        Key in :attr:`~anndata.AnnData.uns` to store the one-hot encodings.
    exclude_values
        Value(s) to exclude from the one-hot encoding.
    copy
        Return a copy of ``adata`` instead of updating it in place.

    Returns
    -------
        If ``copy`` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the one-hot
        encodings stored in :attr:`~anndata.AnnData.uns`. Otherwise, updates ``adata`` in place.

        Sets the following fields:

        - ``.uns[uns_key_added]``: Dictionary containing the one-hot encodings for each covariate.
    """
    adata = adata.copy() if copy else adata

    covariate_keys = _to_list(covariate_keys)
    exclude_values = _to_list(exclude_values)

    # Get unique values from all columns
    all_values = np.unique(adata.obs[covariate_keys].values.flatten())
    values_encode = np.setdiff1d(all_values, exclude_values).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encodings = encoder.fit_transform(values_encode)

    # Store encodings in adata.uns
    adata.uns[uns_key_added] = {}
    for value, encoding in zip(values_encode, encodings, strict=False):
        adata.uns[uns_key_added][value[0]] = encoding

    if copy:
        return adata
