import base64
import requests
import sys
import os


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        print("Skipping", output_fn)
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        # sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.mp4",
        prompt="A deep sea video of a bioluminescent siphonophore, 8k, beautiful, award winning, close up",
        seed=42,
        num_frames=24,
        model="potat1",
        num_inference_steps=30,
        guidance_scale=17.5,
        fps=12,
    )
    gen(
        "vid-sample.mp4",
        prompt="A deep sea video of a bioluminescent siphonophore, 8k, beautiful, award winning, close up",
        seed=42,
        num_frames=24,
        model="zeroscope_v2_XL",
        num_inference_steps=30,
        guidance_scale=17.5,
        init_video="https://replicate.delivery/pbxt/qxacIWhXu0rFAZu6GMElrXrTL5Wx6ZqnjPqIoS7DgIftowkIA/out.mp4",
        fps=12,
    )



if __name__ == "__main__":
    main()
