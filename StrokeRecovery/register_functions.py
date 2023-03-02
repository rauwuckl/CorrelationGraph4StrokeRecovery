# helper code to register functions
# Usage: add the @class_register as a decorator to the class
# add @register('name') as a decorator to each method of the class
# then all functions registered in this way are available under self.registered_functions



def class_register(cls):
    cls._registered_functions = {}

    collector = dict()

    for methodname in dir(cls):
        method = getattr(cls, methodname)
        if hasattr(method, '_name'):
            collector[method._name] = methodname

    old_init = cls.__init__

    def new_init(oself, *args, **kwargs):
        oself.registered_functions = dict()
        for clearname, methodname in collector.items():
            method_pointer = getattr(oself, methodname)
            oself.registered_functions[clearname] = method_pointer
        old_init(oself, *args, **kwargs)

    cls.__init__ = new_init
    return cls

def register(name):
    def wrapper(func):
        func._name = name
        return func
    return wrapper