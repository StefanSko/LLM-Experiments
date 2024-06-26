from modal import Image, Stub, method, gpu, enter

GPU_CONFIG = gpu.A100(memory=40, count=1)
MODEL_REPOS = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
MODEL_FILENAME = "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
MODEL_DIR = "model/"


def download_model():
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=MODEL_REPOS, filename=MODEL_FILENAME, local_dir=MODEL_DIR)

image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .copy_local_file(local_path='json.gbnf', remote_path='./root')
    .pip_install(
        "huggingface_hub",
        "sse_starlette",
    ).run_commands("CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir", gpu=GPU_CONFIG)
    .run_function(download_model)
)

stub = Stub("example-llama-cpp-mixtral", image=image)


@stub.cls(gpu=GPU_CONFIG,
          allow_concurrent_inputs=10,
          container_idle_timeout=20,
          timeout=60,
          image=image)
class Model:

    @enter()
    def startup(self):
        from llama_cpp import Llama, LlamaGrammar
        self.llama = Llama(MODEL_DIR + "/" + MODEL_FILENAME, n_ctx=4096, n_gpu_layers=50, verbose=True)
        #with open('json.gbnf', 'r') as f:
        #    grammar = f.read()
        #self.grammar = LlamaGrammar.from_string(grammar)


    @method()
    def generate(self, text: str):

        prompt = f"""[INST] Please analyze the following text and provide a JSON output with annotations for tonality and keywords:
<text>
{text}
</text>

Generate the output in the following JSON format:

{{
  "tonality": "[Detected tonality]",
  "keywords": [
    "[Keyword 1]",
    "[Keyword 2]",
    ...
  ]
}}

[/INST]"""

        return self.llama(
            prompt,
            max_tokens=2000,
            temperature=0.1,
            echo=False,
            stream=False)

@stub.local_entrypoint()
def main():
    with open('text', 'r') as f:
        text = f.read()
    print(
        Model().generate.remote(text)
    )

