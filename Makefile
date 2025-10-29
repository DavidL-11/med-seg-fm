.PHONY: all clean train submodules venv torch_gpu torch_cpu torch_auto dependencies

all: submodules venv torch_auto dependencies
	@echo "Project setup complete!"

# Pull all submodules
submodules:
	@echo "Pulling all submodules..."
	git submodule update --init --recursive

# Check if NVIDIA GPU is available and set torch installation accordingly
torch_auto: venv
	@echo "Detecting GPU and installing appropriate PyTorch..."
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, installing GPU version of PyTorch..."; \
		$(MAKE) torch_gpu; \
	else \
		echo "No NVIDIA GPU detected, installing CPU version of PyTorch..."; \
		$(MAKE) torch_cpu; \
	fi

venv:
	@echo "Preparing virtual environments..."
	python3 -m venv .venv
	python3 -m venv .venv_med

torch_gpu:
	.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	.venv/bin/pip install onnxruntime-gpu

	.venv_med/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	.venv_med/bin/pip install onnxruntime-gpu

torch_cpu:
	.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install onnxruntime

	.venv_med/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	.venv_med/bin/pip install onnxruntime
	

dependencies: torch_auto
	@echo "Installing dependencies..."
	.venv/bin/pip install matplotlib pyqt6 numpy pandas opencv-python ultralytics nibabel napari
	.venv_med/bin/pip install matplotlib pyqt6 numpy pandas opencv-python vedo ultralytics scikit-learn scikit-image nibabel simpleitk napari
	.venv/bin/pip install -e .
	.venv_med/bin/pip install -e .

	# Install surface distance repo
	.venv/bin/pip install -e src/segFM/predictors/surface-distance
	.venv_med/bin/pip install -e src/segFM/predictors/surface-distance

clean:
	@echo "Cleaning up virtual environments..."
	rm -rf .venv .venv_med

