from src.models.origin.model import model as OriginModel
import torch

def instantiate_model(model_type: str, **kwargs):

    if model_type == 'origin':
        layers = kwargs.get('nlayers', 0)
        if layers <= 0:
            if kwargs['fusion_method'] == 'differential_transformer':
                layers = 8
            elif kwargs['fusion_method'] == 'differential_perceiver':
                layers = 4
            else:
                layers = 8
        return OriginModel(
            ntoken=kwargs['ntoken'],
            d_model=kwargs['d_model'],
            fusion_method=kwargs['fusion_method'],
            attention_backend=kwargs['attention_backend'],
            nlayers=layers,
            perturbation_function=kwargs['perturbation_function'],
            drug_embedding_mode=kwargs['drug_embedding_mode'],
            drug_mechanism_features=kwargs.get('drug_mechanism_features'),
            conditioning_mode=kwargs['conditioning_mode'],
            use_cell_type_embedding=kwargs.get('use_cell_type_embedding', False),
            n_cell_types=kwargs.get('n_cell_types', 0),
            mask_path=kwargs['mask_path'],
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
if __name__ == "__main__":
    model = instantiate_model("punet128")
    x = torch.randn(32,  128, 128)
    t = torch.randn(32)
    out = model( x,t)
    print(out.shape)
