"""Factory for easily getting networks by name."""

from importlib import import_module


def acronym_localizer_case(x):
    acronyms = {
        "baseline": "None",
        "mlp": "FrameMLP",
        "emlp": "FrameEMLP",
        "resmlp": "FrameResMLP",
        "gvp": "FrameGVP",
        "transformer_temporal": "FrameTransformer",
        "spatio_temporal_gat": "SpatioTemporalFrame_GAT",
        "spatio_temporal": "SpatioTemporalFrame",
    }
    # print(x, acronyms.get(x, x))
    return acronyms.get(x, x.capitalize())


# Create a function that loads a pytorch module depending on the localizer type
class LocalizerFactory(object):
    @staticmethod
    def create(module_name, *args, **kwargs):
        class_name = acronym_localizer_case(module_name)
        try:
            model_module = import_module("locs_md.localise")
            module_class = getattr(model_module, class_name)
            module_instance = module_class(*args, **kwargs)
        except (AttributeError, ImportError):
            raise

        return module_instance
