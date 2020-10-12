import importlib


modules = {
    "complexsom": "ComplexSOM",
    "ensemble1": "DEEP_ENSEMBLE_main",
    "ensemble2": "Deep_Ensemble_jax",
    "gpzbinning": "GPz_classifier",
    "lgbm": "LGBM_classifier",
    "lstm": "LSTM_bidirectional",
    "pqnld": "PQNLD",
    "simplesom": "SimpleSOM",
    "tcn": "TCN",
    "utopia": "UTOPIA",
    "autokeras_lstm": "autokeras_LSTM",
    "cnn": "conv_ak",
    "ibandonly": "iband_only",
    "jaxcnn": "jaxCNN",
    "jaxresnet": "jaxResNet",
    "mycombinedclassifiers": "jec_CombineClassifier",
    "flax_lstm": "lstm_flax",
    "minecraft": "minecraft",
    "mlpqna": "mlpqna",
    "funbins": "myclassifier",
    "neuralnetwork": "neural_network",
    "pcacluster": "pca_cluster",
    "randomforest": "random_forest",
    "summerslasher": "summer_slasher",
    "random": "trivial",
    "zotbin": "zotbin",
    "zotnet": "zotnet",
}



class Tomographer:
    _subclasses = {}
    wants_arrays = False
    skips_zero_flux = False

    @classmethod
    def _find_subclass(cls, name):
        try:
            module = modules[name.lower()]
            importlib.import_module(f'tomo_challenge.classifiers.{module}')
        except KeyError:
            raise ValueError(f"Unknown module {module}")
        return cls._subclasses[name]

    def __init_subclass__(cls, *args, **kwargs):
        module = cls.__module__.split('.')[-1]
        print(f"Found classifier {cls.__name__} in {module}")

        if cls.__name__ in cls._subclasses:
            raise ValueError(f"Duplicate name: {cls.__name__}")
        cls._subclasses[cls.__name__] = cls
