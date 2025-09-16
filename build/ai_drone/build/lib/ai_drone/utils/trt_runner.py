import tensorrt as trt, pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

class TrtRunner:
    """Charge un moteur TensorRT et exécute une inférence sur entrée NCHW float32 [1,3,H,W]."""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path,'rb') as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.bindings=[None]*self.engine.num_bindings
        self.host=[] 
        self.dev=[] 
        self.stream=cuda.Stream()
        for i in range(self.engine.num_bindings):
            shp = self.engine.get_binding_shape(i)
            dt  = trt.nptype(self.engine.get_binding_dtype(i))
            size= int(np.prod(shp))
            h = cuda.pagelocked_empty(size, dt)
            d = cuda.mem_alloc(h.nbytes)
            self.host.append(h) 
            self.dev.append(d)
            self.bindings[i]=int(d)

    def infer(self, x: np.ndarray) -> list:
        """Retourne une liste de sorties numpy à partir du tenseur d’entrée x."""
        np.copyto(self.host[0], x.ravel())
        cuda.memcpy_htod_async(self.dev[0], self.host[0], self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        for i in range(1, self.engine.num_bindings):
            cuda.memcpy_dtoh_async(self.host[i], self.dev[i], self.stream)
        self.stream.synchronize()
        return [np.array(self.host[i]) for i in range(1, self.engine.num_bindings)]

