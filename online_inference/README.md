To build the docker image:
~~~
docker build . -t skrivva/online_inference_image
~~~

To run docker and get model predictions:
~~~
docker pull skrivva/online_inference_image
docker run -p 8000:8000 skrivva/online_inference_image
py make_request.py
~~~