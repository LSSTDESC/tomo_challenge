class Tomographer:
    _subclasses = {}
    wants_arrays = False

    @classmethod
    def _find_subclass(cls, name):
        return cls._subclasses[name]

    def __init_subclass__(cls, *args, **kwargs):
        print(f"Found classifier {cls.__name__}")
        cls._subclasses[cls.__name__] = cls
