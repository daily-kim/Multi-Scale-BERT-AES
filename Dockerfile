
FROM nvcr.io/nvidia/pytorch:22.02-py3
# Upgrade pip
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install git -y

RUN mkdir -p /workspace 
WORKDIR /workspace 

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import torch; print(torch.__version__)"

RUN python -c "import torch; print(torch.cuda.is_available())"

