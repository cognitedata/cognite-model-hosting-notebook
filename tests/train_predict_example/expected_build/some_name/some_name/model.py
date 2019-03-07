# !notebook-cell
def train_model(open_artifact, data_spec):
    pass


# !notebook-cell
def load_model(open_artifact):
    pass


# !notebook-cell
def predict(model, instance, some_argument):
    pass


# !auto-generated
class Model:
    def __init__(self, model):
        self._model = model

    @staticmethod
    def train(open_artifact, data_spec):
        train_model(open_artifact, data_spec)

    @staticmethod
    def load(open_artifact):
        return Model(load_model(open_artifact))

    def predict(self, instance, some_argument):
        return predict(self._model, instance, some_argument)
