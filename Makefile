default: build

help:
	@echo 'Management commands for msbert:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t msbert 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name msbert -v `pwd`:/workspace msbert:latest /bin/bash

up: build run

rm: 
	@docker rm msbert

stop:
	@docker stop msbert

reset: stop rm
