# API

<!-- vim-markdown-toc GFM -->

* [List Models](#list-models)
  * [Request](#request)
  * [Response](#response)
* [Generate request (Streaming)](#generate-request-streaming)
  * [Request](#request-1)
  * [Response](#response-1)
* [Request (No streaming)](#request-no-streaming)
  * [Request](#request-2)
  * [Response](#response-2)
* [Python](#python)
  * [Install](#install)
  * [Usage](#usage)
  * [Output](#output)
  * [List of models](#list-of-models)
    * [Code](#code)
    * [Output](#output-1)

<!-- vim-markdown-toc -->

## List Models

### Request

```shell
curl http://10.204.100.72:11434/api/tags
```

### Response

A stream of JSON objects is returned:

```json
{
  "models": [
    {
      "name": "llama3:70b-instruct",
      "model": "llama3:70b-instruct",
      "modified_at": "2024-05-06T03:53:28.90499374Z",
      "size": 39969745251,
      "digest": "be39eb53a197ec3a34aab3b4b628169e61f2f28c350d51995744d8ec0f3e6747",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "71B",
        "quantization_level": "Q4_0"
      }
    },
    {
      "name": "llama3:70b-instruct-fp16",
      "model": "llama3:70b-instruct-fp16",
      "modified_at": "2024-04-30T04:47:31Z",
      "size": 141117925698,
      "digest": "49a263bc03b9a5cb9dd33f22655d0885b7d19bdce7418cfe5038873227f3d7d0",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "71B",
        "quantization_level": "F16"
      }
    },
    {
      "name": "llama3:70b-instruct-q8_0",
      "model": "llama3:70b-instruct-q8_0",
      "modified_at": "2024-05-06T04:12:44.698809162Z",
      "size": 74975062371,
      "digest": "d6fa8cffc283faf3dfa501a3cdcfc805db4cab013b1d21b245ad297603fe6bda",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "71B",
        "quantization_level": "Q8_0"
      }
    }
  ]
}
```

## Generate request (Streaming)

### Request

```shell
curl http://10.204.100.72:11434/api/generate -d '{
  "model": "llama3:70b-instruct",
  "prompt": "Why is the sky blue?"
}'
```

### Response

A stream of JSON objects is returned:

```json
{
  "model": "llama3:70b-instruct",
  "created_at": "2024-05-06T04:26:56.970880916Z",
  "response": "The",
  "done": false
}
{
  "model":"llama3:70b-instruct",
  "created_at":"2024-05-06T04:26:57.014303471Z",
  "response":" sky",
  "done":false
}
{
  "model":"llama3:70b-instruct",
  "created_at":"2024-05-06T04:26:57.057275692Z",
  "response":" appears",
  "done":false
}

...

{
  "model": "llama3:70b-instruct",
  "created_at": "2024-05-06T04:27:10.667656536Z",
  "response": "",
  "done": true,
  "context": [
    128006, 882, 128007, 198, 198, 10445, 374, 279, 13180, 6437, 30, 128009,
    128006, 78191, 128007, 198, 198, 791, 13180, 8111, 6437, 1606, 315, 264,
    25885, 2663, 13558, 64069, 72916, 11, 902, 374, 279, 72916, 315, 3177, 555,
    2678, 19252, 477, 35715, 304, 279, 16975, 627, 198, 8586, 596, 1148, 8741,
    512, 198, 16, 13, 3146, 31192, 4238, 29933, 9420, 596, 16975, 96618, 3277,
    40120, 29933, 1057, 16975, 11, 433, 35006, 13987, 35715, 315, 45612, 1093,
    47503, 320, 45, 17, 8, 323, 24463, 320, 46, 17, 570, 4314, 35715, 527, 1790,
    9333, 1109, 279, 46406, 315, 3177, 13, 198, 17, 13, 3146, 3407, 31436,
    13980, 96618, 578, 24210, 320, 12481, 8, 93959, 315, 3177, 527, 38067, 304,
    682, 18445, 555, 1521, 2678, 35715, 11, 1418, 279, 5129, 320, 1171, 8,
    93959, 3136, 311, 5944, 304, 264, 7833, 1584, 449, 2697, 72916, 13, 1115,
    374, 3967, 439, 13558, 64069, 72916, 13, 198, 18, 13, 3146, 10544, 3177,
    374, 77810, 96618, 1666, 264, 1121, 315, 420, 72916, 11, 279, 6437, 3177,
    374, 77810, 6957, 279, 16975, 11, 19261, 1057, 6548, 505, 682, 18445, 13,
    198, 19, 13, 3146, 1687, 1518, 279, 13180, 439, 6437, 96618, 8876, 584,
    1518, 279, 6437, 3177, 1694, 38067, 304, 1475, 5216, 11, 279, 13180, 8111,
    6437, 311, 603, 627, 198, 2028, 2515, 374, 810, 38617, 2391, 279, 62182,
    994, 279, 7160, 374, 32115, 11, 323, 279, 3392, 315, 72916, 12992, 449,
    36958, 13, 3011, 596, 3249, 279, 13180, 11383, 8111, 810, 19428, 6437, 520,
    5190, 12231, 811, 627, 198, 2181, 596, 1101, 5922, 27401, 430, 45475, 4787,
    1093, 25793, 11, 16174, 11, 323, 3090, 38752, 649, 45577, 3177, 304, 2204,
    5627, 11, 902, 649, 2349, 279, 10186, 1933, 315, 279, 13180, 13, 1789, 3187,
    11, 2391, 64919, 323, 44084, 11, 279, 13180, 3629, 5097, 389, 82757, 315,
    2579, 11, 19087, 11, 323, 18718, 4245, 311, 279, 72916, 315, 3177, 555,
    45475, 19252, 627, 198, 4516, 11, 1070, 499, 617, 433, 0, 578, 13180, 8111,
    6437, 1606, 315, 279, 72916, 315, 40120, 555, 13987, 35715, 304, 1057,
    16975, 13, 128009
  ],
  "total_duration": 31995338983,
  "load_duration": 17662231021,
  "prompt_eval_count": 16,
  "prompt_eval_duration": 595458000,
  "eval_count": 316,
  "eval_duration": 13696729000
}
```

The final response in the stream also includes additional data about the generation:

- `total_duration`: time spent generating the response
- `load_duration`: time spent in nanoseconds loading the model
- `prompt_eval_count`: number of tokens in the prompt
- `prompt_eval_duration`: time spent in nanoseconds evaluating the prompt
- `eval_count`: number of tokens in the response
- `eval_duration`: time in nanoseconds spent generating the response
- `context`: an encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory
- `response`: empty if the response was streamed, if not streamed, this will contain the full response

## Request (No streaming)

### Request

A response can be received in one reply when streaming is off.

```shell
curl http://10.204.100.72:11434/api/generate -d '{
  "model": "llama3:70b-instruct",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Response

If `stream` is set to `false`, the response will be a single JSON object:

```json
{
  "model": "llama3:70b-instruct",
  "created_at": "2024-05-06T04:40:12.198251921Z",
  "response": "The sky appears blue because of a phenomenon called Rayleigh scattering, which is the scattering of light by small particles or molecules in the atmosphere. Here's a simplified explanation:\n\n1. **Sunlight enters the Earth's atmosphere**: When sunlight enters our atmosphere, it encounters tiny molecules of gases like nitrogen (N2) and oxygen (O2).\n2. **Short wavelengths are scattered more**: These gas molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths. This is known as Rayleigh scattering.\n3. **Blue light is scattered in all directions**: As a result of this scattering, the blue light is distributed throughout the atmosphere, reaching our eyes from all parts of the sky.\n4. **Red light continues to travel in a straight line**: The longer wavelengths of red light are not scattered as much and continue to travel in a more direct path to our eyes, appearing as a beam of light.\n5. **Our eyes perceive the blue light as the sky's color**: Since we see the scattered blue light coming from all directions, our brains interpret it as the color of the sky.\n\nThis is why the sky typically appears blue during the daytime, especially in the direction away from the sun. The exact shade of blue can vary depending on factors like:\n\n* Atmospheric conditions (e.g., pollution, dust, water vapor)\n* Time of day and year (e.g., sunrise/sunset, seasonal changes)\n* Altitude and atmospheric pressure\n* Cloud cover and type\n\nSo, to summarize, the sky appears blue because of the scattering of sunlight by tiny molecules in the atmosphere, which favors shorter wavelengths like blue light.",
  "done": true,
  "context": [
    128006, 882, 128007, 198, 198, 10445, 374, 279, 13180, 6437, 30, 128009,
    128006, 78191, 128007, 198, 198, 791, 13180, 8111, 6437, 1606, 315, 264,
    25885, 2663, 13558, 64069, 72916, 11, 902, 374, 279, 72916, 315, 3177, 555,
    2678, 19252, 477, 35715, 304, 279, 16975, 13, 5810, 596, 264, 44899, 16540,
    512, 198, 16, 13, 3146, 31192, 4238, 29933, 279, 9420, 596, 16975, 96618,
    3277, 40120, 29933, 1057, 16975, 11, 433, 35006, 13987, 35715, 315, 45612,
    1093, 47503, 320, 45, 17, 8, 323, 24463, 320, 46, 17, 570, 198, 17, 13,
    3146, 12755, 93959, 527, 38067, 810, 96618, 4314, 6962, 35715, 45577, 279,
    3177, 304, 682, 18445, 11, 719, 814, 45577, 24210, 320, 12481, 8, 93959,
    810, 1109, 5129, 320, 1171, 8, 93959, 13, 1115, 374, 3967, 439, 13558,
    64069, 72916, 13, 198, 18, 13, 3146, 10544, 3177, 374, 38067, 304, 682,
    18445, 96618, 1666, 264, 1121, 315, 420, 72916, 11, 279, 6437, 3177, 374,
    4332, 6957, 279, 16975, 11, 19261, 1057, 6548, 505, 682, 5596, 315, 279,
    13180, 13, 198, 19, 13, 3146, 6161, 3177, 9731, 311, 5944, 304, 264, 7833,
    1584, 96618, 578, 5129, 93959, 315, 2579, 3177, 527, 539, 38067, 439, 1790,
    323, 3136, 311, 5944, 304, 264, 810, 2167, 1853, 311, 1057, 6548, 11, 26449,
    439, 264, 24310, 315, 3177, 13, 198, 20, 13, 3146, 8140, 6548, 45493, 279,
    6437, 3177, 439, 279, 13180, 596, 1933, 96618, 8876, 584, 1518, 279, 38067,
    6437, 3177, 5108, 505, 682, 18445, 11, 1057, 35202, 14532, 433, 439, 279,
    1933, 315, 279, 13180, 627, 198, 2028, 374, 3249, 279, 13180, 11383, 8111,
    6437, 2391, 279, 62182, 11, 5423, 304, 279, 5216, 3201, 505, 279, 7160, 13,
    578, 4839, 28601, 315, 6437, 649, 13592, 11911, 389, 9547, 1093, 1473, 9,
    87597, 4787, 320, 68, 13, 70, 2637, 25793, 11, 16174, 11, 3090, 38752, 8,
    198, 9, 4212, 315, 1938, 323, 1060, 320, 68, 13, 70, 2637, 64919, 2754,
    37904, 11, 36899, 4442, 8, 198, 9, 24610, 3993, 323, 45475, 7410, 198, 9,
    15161, 3504, 323, 955, 198, 198, 4516, 11, 311, 63179, 11, 279, 13180, 8111,
    6437, 1606, 315, 279, 72916, 315, 40120, 555, 13987, 35715, 304, 279, 16975,
    11, 902, 54947, 24210, 93959, 1093, 6437, 3177, 13, 128009
  ],
  "total_duration": 32491985634,
  "load_duration": 17126873274,
  "prompt_eval_count": 16,
  "prompt_eval_duration": 578557000,
  "eval_count": 339,
  "eval_duration": 14746345000
}
```

## Python

### Install

```sh
pip install ollama
```

### Usage

```python
from ollama import Client

client = Client(host="http://10.204.100.72:11434")
response = client.chat(
    model="llama3:70b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ],
)
print(response["message"])
```

### Output

```python
{
    "role": "assistant",
    "content": "The sky appears blue because of a phenomenon called Rayleigh scattering, which is the scattering of light by small particles or molecules in the atmosphere. Here's a simplified explanation:\n\n1. **Sunlight enters Earth's atmosphere**: When sunlight enters our planet's atmosphere, it encounters tiny molecules of gases like nitrogen (N2) and oxygen (O2).\n2. **Scattering occurs**: These gas molecules scatter the shorter (blue) wavelengths of light more than the longer (red) wavelengths. This is known as Rayleigh scattering.\n3. **Blue light is scattered in all directions**: As a result, the blue light is dispersed throughout the atmosphere, reaching our eyes from all parts of the sky.\n4. **Red light continues its path**: Meanwhile, the longer wavelengths of light, like red and orange, are not scattered as much and continue to travel in a more direct path to our eyes.\n5. **Our brains perceive the blue color**: Since we see the scattered blue light coming from all directions, our brains interpret this as the sky being blue.\n\nThis effect is more pronounced during the daytime when the sun is overhead, which is why the sky typically appears more blue then. At sunrise and sunset, when the sun's angle is lower, the light has to travel through more of the atmosphere, scattering off even more molecules, which is why we often see more red and orange hues during these times.\n\nIt's worth noting that atmospheric conditions like pollution, dust, and water vapor can affect the color of the sky, making it appear more hazy or gray. However, in general, the blue color of the sky is a result of Rayleigh scattering and the way our atmosphere interacts with sunlight.",
}
```

### List of models

#### Code

```python
client.list()
```

#### Output

```python
{
    "models": [
        {
            "name": "llama3:70b-instruct",
            "model": "llama3:70b-instruct",
            "modified_at": "2024-05-06T03:53:28.90499374Z",
            "size": 39969745251,
            "digest": "be39eb53a197ec3a34aab3b4b628169e61f2f28c350d51995744d8ec0f3e6747",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "71B",
                "quantization_level": "Q4_0",
            },
        },
        {
            "name": "llama3:70b-instruct-fp16",
            "model": "llama3:70b-instruct-fp16",
            "modified_at": "2024-04-30T04:47:31Z",
            "size": 141117925698,
            "digest": "49a263bc03b9a5cb9dd33f22655d0885b7d19bdce7418cfe5038873227f3d7d0",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "71B",
                "quantization_level": "F16",
            },
        },
        {
            "name": "llama3:70b-instruct-q8_0",
            "model": "llama3:70b-instruct-q8_0",
            "modified_at": "2024-05-06T04:12:44.698809162Z",
            "size": 74975062371,
            "digest": "d6fa8cffc283faf3dfa501a3cdcfc805db4cab013b1d21b245ad297603fe6bda",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "71B",
                "quantization_level": "Q8_0",
            },
        },
    ]
}
```
