# Use the official AWS Lambda Python base image. It includes the Lambda Runtime Interface Client.
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory in the container
WORKDIR /var/task

# Copy requirements.txt first to leverage Docker's layer caching.
# If requirements.txt hasn't changed, this layer won't be rebuilt.
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py ./

# Set the command to your handler.
# The format is "<script_name>.<handler_function_name>"
# In our case, it's the `handler` object created by Mangum in `app.py`.
CMD [ "app.handler" ]
