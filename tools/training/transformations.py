from torch_geometric.transforms import BaseTransform, Compose


def _safe_standardize(values, eps=1e-8):
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(eps)
    return (values - mean) / std

class NormalizeNodeFeatures(BaseTransform):
    def forward(self, data):
        if hasattr(data, 'x'):
            data.x = _safe_standardize(data.x)
        return data

class NormalizeEdgeFeatures(BaseTransform):
    def forward(self, data):
        if hasattr(data, 'edge_attr'):
            data.edge_attr = _safe_standardize(data.edge_attr)
        return data

class NormalizeTargets(BaseTransform):
    def forward(self, data):
        if hasattr(data, 'y'):
            data.y = _safe_standardize(data.y)
        return data

class DropLastTwoNodeFeatures(BaseTransform):
    def forward(self, data):
        if hasattr(data, 'x'):
            data.x = data.x[:, :-2]  # Eliminar las dos últimas columnas
        return data

class NormalizeSpecificNodeFeatures(BaseTransform):
    def __init__(self, column_indices):
        self.column_indices = column_indices

    def forward(self, data):
        if hasattr(data, 'x'):
            for column_index in self.column_indices:
                column = data.x[:, column_index]
                mean = column.mean()
                std = column.std(unbiased=False).clamp_min(1e-8)
                data.x[:, column_index] = (column - mean) / std
        return data

# Definir las transformaciones
NormalizeNodeEdgesAndDropTwoFeatures = Compose([
    NormalizeNodeFeatures(),
    NormalizeEdgeFeatures(),
    DropLastTwoNodeFeatures()  # Aplicar la transformación para eliminar las dos últimas características
])
