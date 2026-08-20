"""
Low-level AscendCL model wrapper.

The class loads a single OM model, manages input/output device buffers, and
exposes a numpy-friendly ``infer`` method. Dynamic-shape models (inputs named
``ascend_mbatch_shape_data``) are detected automatically and the HW size is
updated before each execution.
"""

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from .acl_env import acl_get_context, acl_init, acl_release
from .exceptions import InferenceError, ModelLoadError, YuntuAscendOCRError

try:
    import acl
except ImportError:  # pragma: no cover - only available on Ascend hardware
    acl = None


logger = logging.getLogger(__name__)

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


class AscendModel:
    """Wrapper around a single Ascend OM offline model."""

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        decrypt_callback: Optional[Callable[[str], bytes]] = None,
    ):
        """
        Args:
            model_path: Path to the ``.om`` model file. May be encrypted if
                ``decrypt_callback`` is supplied.
            device_id: Ascend NPU device id.
            decrypt_callback: Optional callable ``f(path) -> bytes`` that returns
                the decrypted model bytes. When omitted the file is loaded directly.
        """
        if acl is None:
            raise YuntuAscendOCRError(
                "The 'acl' package is not available on this machine."
            )

        self.model_path = model_path
        self.device_id = device_id
        self.decrypt_callback = decrypt_callback
        self._released = False

        # Initialize ACL context.
        acl_init(device_id)

        # Load model.
        self.model_id = self._load_model()

        # Build model descriptor.
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise ModelLoadError(
                f"acl.mdl.get_desc failed for {model_path}, ret={ret}"
            )

        self._dynamic_input = self._detect_dynamic_input()
        self._input_num = acl.mdl.get_num_inputs(self.model_desc)
        self._output_num = acl.mdl.get_num_outputs(self.model_desc)
        self._print_model_desc()

        # Create input/output datasets.
        self.input_dataset, self.input_data = self._prepare_dataset("input")
        self.output_dataset, self.output_data = self._prepare_dataset("output")

    def _load_model(self) -> int:
        if not os.path.exists(self.model_path):
            raise ModelLoadError(f"Model file not found: {self.model_path}")

        if self.decrypt_callback is not None:
            logger.info("Loading encrypted model from memory: %s", self.model_path)
            plain_bytes = self.decrypt_callback(self.model_path)
            model_ptr = acl.util.bytes_to_ptr(plain_bytes)
            model_id, ret = acl.mdl.load_from_mem(model_ptr, len(plain_bytes))
        else:
            logger.info("Loading model from file: %s", self.model_path)
            model_id, ret = acl.mdl.load_from_file(self.model_path)

        if ret != 0:
            raise ModelLoadError(
                f"Failed to load model {self.model_path}, ret={ret}"
            )
        return model_id

    def _detect_dynamic_input(self) -> bool:
        input_num = acl.mdl.get_num_inputs(self.model_desc)
        for i in range(input_num):
            try:
                name = acl.mdl.get_input_name_by_index(self.model_desc, i)
                if name == "ascend_mbatch_shape_data":
                    return True
            except Exception:
                pass
        return False

    def _print_model_desc(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            input_num = acl.mdl.get_num_inputs(self.model_desc)
            output_num = acl.mdl.get_num_outputs(self.model_desc)
            logger.debug("Model %s:", self.model_path)
            logger.debug("  dynamic HW: %s", self._dynamic_input)
            logger.debug("  inputs: %d", input_num)
            for i in range(input_num):
                size = acl.mdl.get_input_size_by_index(self.model_desc, i)
                name = acl.mdl.get_input_name_by_index(self.model_desc, i)
                dims, _ = acl.mdl.get_input_dims(self.model_desc, i)
                logger.debug(
                    "    input[%d] name=%s size=%d dims=%s",
                    i, name, size, dims.get("dims", dims),
                )
            logger.debug("  outputs: %d", output_num)
            for i in range(output_num):
                size = acl.mdl.get_output_size_by_index(self.model_desc, i)
                name = acl.mdl.get_output_name_by_index(self.model_desc, i)
                dims, _ = acl.mdl.get_output_dims(self.model_desc, i)
                logger.debug(
                    "    output[%d] name=%s size=%d dims=%s",
                    i, name, size, dims.get("dims", dims),
                )
        except Exception as exc:
            logger.debug("Failed to print model desc: %s", exc)

    def _prepare_dataset(self, io_type: str):
        dataset = acl.mdl.create_dataset()
        if io_type == "input":
            io_num = self._input_num
            get_size = acl.mdl.get_input_size_by_index
        else:
            io_num = self._output_num
            get_size = acl.mdl.get_output_size_by_index

        datas = []
        for i in range(io_num):
            buffer_size = get_size(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != 0:
                raise ModelLoadError(
                    f"acl.rt.malloc failed for {io_type}[{i}], ret={ret}"
                )
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def _set_dynamic_hw(self, inputs: List[np.ndarray]) -> None:
        """Update dynamic HW size for the first input tensor."""
        if not self._dynamic_input:
            return
        arr = inputs[0]
        if len(arr.shape) == 4:
            height, width = arr.shape[2], arr.shape[3]
        elif len(arr.shape) == 3:
            height, width = arr.shape[1], arr.shape[2]
        else:
            return

        index, ret = acl.mdl.get_input_index_by_name(
            self.model_desc, "ascend_mbatch_shape_data"
        )
        if ret != 0:
            raise InferenceError(
                f"acl.mdl.get_input_index_by_name failed, ret={ret}"
            )
        ret = acl.mdl.set_dynamic_hw_size(
            self.model_id, self.input_dataset, index, height, width
        )
        if ret != 0:
            raise InferenceError(
                f"acl.mdl.set_dynamic_hw_size failed, ret={ret}"
            )

    def infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on the given numpy inputs.

        Args:
            inputs: List of numpy arrays matching the model input order.

        Returns:
            List of numpy arrays containing model outputs.
        """
        if self._released:
            raise InferenceError("Model has already been released")

        # Set current context.
        ret = acl.rt.set_context(acl_get_context(self.device_id))
        if ret != 0:
            raise InferenceError(f"acl.rt.set_context failed, ret={ret}")

        # Validate input count.
        if len(inputs) != self._input_num:
            raise InferenceError(
                f"Expected {self._input_num} inputs, got {len(inputs)}"
            )

        # Copy inputs to device.
        for i, arr in enumerate(inputs):
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            bytes_data = arr.tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            buffer_size = self.input_data[i]["size"]
            if len(bytes_data) > buffer_size:
                raise InferenceError(
                    f"Input[{i}] size {len(bytes_data)} exceeds buffer {buffer_size}"
                )
            if len(bytes_data) != buffer_size:
                logger.warning(
                    "Input[%d] size %d != model buffer %d (shape %s): "
                    "the model will read uninitialized memory. Check that "
                    "preprocessing matches the model input shape.",
                    i, len(bytes_data), buffer_size, arr.shape,
                )
            ret = acl.rt.memcpy(
                self.input_data[i]["buffer"],
                buffer_size,
                bytes_ptr,
                len(bytes_data),
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
            if ret != 0:
                raise InferenceError(
                    f"acl.rt.memcpy input[{i}] failed, ret={ret}"
                )

        # Update dynamic shape if needed.
        self._set_dynamic_hw(inputs)

        logger.debug(
            "infer %s: input shapes=%s",
            os.path.basename(self.model_path),
            [list(a.shape) for a in inputs],
        )

        # Execute.
        ret = acl.mdl.execute(
            self.model_id, self.input_dataset, self.output_dataset
        )
        if ret != 0:
            raise InferenceError(f"acl.mdl.execute failed, ret={ret}")

        # Copy outputs to host.
        outputs = []
        for i in range(self._output_num):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            if ret != 0:
                raise InferenceError(
                    f"acl.rt.malloc_host output[{i}] failed, ret={ret}"
                )
            ret = acl.rt.memcpy(
                buffer_host,
                self.output_data[i]["size"],
                self.output_data[i]["buffer"],
                self.output_data[i]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            if ret != 0:
                acl.rt.free_host(buffer_host)
                raise InferenceError(
                    f"acl.rt.memcpy output[{i}] failed, ret={ret}"
                )

            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            out_dim = dims["dims"]
            bytes_out = acl.util.ptr_to_bytes(
                buffer_host, self.output_data[i]["size"]
            )
            acl.rt.free_host(buffer_host)

            # DBNet and recognition models both output float32.
            data = np.frombuffer(bytes_out, dtype=np.float32).reshape(out_dim)
            outputs.append(data)
        logger.debug(
            "infer %s: output shapes=%s",
            os.path.basename(self.model_path),
            [list(o.shape) for o in outputs],
        )
        return outputs

    @property
    def dynamic_input(self) -> bool:
        """Return True if the model has a dynamic HW input."""
        return self._dynamic_input

    def get_input_shape(self, index: int = 0) -> Tuple[int, ...]:
        """Return the static input shape declared by the model."""
        dims, _ = acl.mdl.get_input_dims(self.model_desc, index)
        return tuple(dims["dims"])

    def get_output_shape(self, index: int = 0) -> Tuple[int, ...]:
        """Return the static output shape declared by the model."""
        dims, _ = acl.mdl.get_output_dims(self.model_desc, index)
        return tuple(dims["dims"])

    def release(self) -> None:
        """Release all ACL resources held by this model."""
        if self._released or acl is None:
            return
        try:
            for dataset in [self.input_data, self.output_data]:
                while dataset:
                    item = dataset.pop()
                    acl.destroy_data_buffer(item["data"])
                    acl.rt.free(item["buffer"])
            acl.mdl.destroy_dataset(self.input_dataset)
            acl.mdl.destroy_dataset(self.output_dataset)
            acl.mdl.destroy_desc(self.model_desc)
            acl.mdl.unload(self.model_id)
        finally:
            acl_release(self.device_id)
            self._released = True

    def __del__(self):
        self.release()
