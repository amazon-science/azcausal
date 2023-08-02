FROM public.ecr.aws/lambda/python:3.10



COPY azcausal-0.1.2.tar.gz /tmp
RUN pip3 install /tmp/azcausal-*.tar.gz --target "${LAMBDA_TASK_ROOT}"

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt --target "${LAMBDA_TASK_ROOT}"


COPY handler.py ${LAMBDA_TASK_ROOT}

CMD [ "handler.lambda_handler" ]
