import onnx
import numpy as np
from onnx import numpy_helper


class NumpyONNX:
    def __init__(self, model_path):
        model = onnx.load(model_path)
        self.graph = model.graph
        self.weights = {}
        for init in self.graph.initializer:
            self.weights[init.name] = numpy_helper.to_array(init).astype(np.float32)

    def run(self, x):
        tensors = dict(self.weights)
        tensors[self.graph.input[0].name] = x.astype(np.float32)
        for node in self.graph.node:
            self._run_node(node, tensors)
        return tensors[self.graph.output[0].name]

    def _attr(self, node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                if a.type == 1: return a.f
                if a.type == 2: return a.i
                if a.type == 6: return list(a.floats)
                if a.type == 7: return list(a.ints)
                if a.type == 4: return numpy_helper.to_array(a.t)
        return default

    def _inputs(self, node, tensors):
        return [tensors.get(i) for i in node.input]

    def _run_node(self, node, tensors):
        op = node.op_type
        inp = self._inputs(node, tensors)

        if op == 'Conv':
            out = self._conv(node, inp)
        elif op == 'BatchNormalization':
            x, scale, bias, mean, var = inp[:5]
            eps = self._attr(node, 'epsilon', 1e-5)
            x_norm = (x - mean.reshape(1,-1,1,1)) / np.sqrt(var.reshape(1,-1,1,1) + eps)
            out = scale.reshape(1,-1,1,1) * x_norm + bias.reshape(1,-1,1,1)
        elif op == 'Relu':
            out = np.maximum(inp[0], 0)
        elif op == 'Sigmoid':
            out = 1.0 / (1.0 + np.exp(-np.clip(inp[0], -88, 88)))
        elif op == 'HardSwish':
            x = inp[0]
            out = x * np.clip(x + 3.0, 0.0, 6.0) / 6.0
        elif op == 'Mul':
            out = inp[0] * inp[1]
        elif op == 'Add':
            out = inp[0] + inp[1]
        elif op == 'Sub':
            out = inp[0] - inp[1]
        elif op == 'Div':
            out = inp[0] / inp[1]
        elif op == 'Pow':
            out = np.power(inp[0], inp[1])
        elif op == 'Sqrt':
            out = np.sqrt(inp[0])
        elif op == 'Clip':
            lo = float(inp[1]) if len(inp) > 1 and inp[1] is not None else self._attr(node, 'min', None)
            hi = float(inp[2]) if len(inp) > 2 and inp[2] is not None else self._attr(node, 'max', None)
            out = np.clip(inp[0], lo, hi)
        elif op == 'GlobalAveragePool':
            out = inp[0].mean(axis=(2, 3), keepdims=True)
        elif op == 'Flatten':
            axis = self._attr(node, 'axis', 1)
            s = inp[0].shape
            out = inp[0].reshape(int(np.prod(s[:axis])), int(np.prod(s[axis:])))
        elif op == 'Reshape':
            shape = inp[1].astype(int).tolist()
            out = inp[0].reshape(shape)
        elif op == 'Transpose':
            out = np.transpose(inp[0], self._attr(node, 'perm'))
        elif op == 'Gemm':
            A, B = inp[0], inp[1]
            C = inp[2] if len(inp) > 2 and inp[2] is not None else 0
            if self._attr(node, 'transA', 0): A = A.T
            if self._attr(node, 'transB', 0): B = B.T
            out = self._attr(node, 'alpha', 1.0) * np.dot(A, B) + self._attr(node, 'beta', 1.0) * C
        elif op == 'MatMul':
            out = np.matmul(inp[0], inp[1])
        elif op == 'Softmax':
            axis = self._attr(node, 'axis', -1)
            e = np.exp(inp[0] - inp[0].max(axis=axis, keepdims=True))
            out = e / e.sum(axis=axis, keepdims=True)
        elif op == 'Dropout':
            out = inp[0]
        elif op == 'Constant':
            out = self._attr(node, 'value')
        elif op == 'Shape':
            out = np.array(inp[0].shape, dtype=np.int64)
        elif op == 'Gather':
            out = np.take(inp[0], inp[1].astype(int), axis=self._attr(node, 'axis', 0))
        elif op == 'Unsqueeze':
            axes = self._attr(node, 'axes') or (inp[1].astype(int).tolist() if len(inp) > 1 and inp[1] is not None else [])
            out = inp[0]
            for ax in sorted(axes):
                out = np.expand_dims(out, axis=ax)
        elif op == 'Squeeze':
            axes = self._attr(node, 'axes') or (inp[1].astype(int).tolist() if len(inp) > 1 and inp[1] is not None else None)
            out = np.squeeze(inp[0], axis=tuple(axes)) if axes else np.squeeze(inp[0])
        elif op == 'Concat':
            out = np.concatenate([i for i in inp if i is not None], axis=self._attr(node, 'axis', 0))
        elif op == 'Cast':
            dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}
            out = inp[0].astype(dtype_map.get(self._attr(node, 'to', 1), np.float32))
        elif op == 'ReduceMean':
            axes = self._attr(node, 'axes')
            out = inp[0].mean(axis=tuple(axes) if axes else None,
                              keepdims=bool(self._attr(node, 'keepdims', 1)))
        elif op == 'Expand':
            out = np.broadcast_to(inp[0], inp[1].astype(int).tolist()).copy()
        else:
            raise NotImplementedError(f'Unsupported op: {op}')

        if node.output:
            tensors[node.output[0]] = out

    def _conv(self, node, inp):
        x, w = inp[0], inp[1]
        b = inp[2] if len(inp) > 2 and inp[2] is not None else None
        groups   = self._attr(node, 'group', 1)
        pads     = self._attr(node, 'pads', [0,0,0,0])
        strides  = self._attr(node, 'strides', [1,1])

        if any(p > 0 for p in pads):
            x = np.pad(x, ((0,0),(0,0),(pads[0],pads[2]),(pads[1],pads[3])))

        N, C, H, W = x.shape
        F, C_g, kH, kW = w.shape
        oH = (H - kH) // strides[0] + 1
        oW = (W - kW) // strides[1] + 1

        # im2col via stride tricks
        xs = np.lib.stride_tricks.as_strided(
            x,
            shape=(N, C, oH, oW, kH, kW),
            strides=(x.strides[0], x.strides[1],
                     x.strides[2]*strides[0], x.strides[3]*strides[1],
                     x.strides[2], x.strides[3])
        )

        out = np.zeros((N, F, oH, oW), dtype=np.float32)
        cpg = C // groups
        fpg = F // groups

        for g in range(groups):
            out[:, g*fpg:(g+1)*fpg] = np.einsum(
                'ncijkl,fckl->nfij',
                xs[:, g*cpg:(g+1)*cpg],
                w[g*fpg:(g+1)*fpg],
                optimize=True
            )

        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        return out
