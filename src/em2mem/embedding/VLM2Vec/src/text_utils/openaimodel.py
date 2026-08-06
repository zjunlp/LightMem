import openai
from time import sleep


class OpenAIModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def completions(self,
                    prompt: str,
                    n: int,
                    temperature: float,
                    top_p: float,
                    max_tokens: int,
                    **kwargs):
        for i in range(100):
            try:
                sleep(1. / 5)
                return self.sample_openai_call(prompt=prompt,
                                               num_return_sequences=n,
                                               temperature=temperature, top_p=top_p,
                                               max_gen_length=max_tokens)
            except Exception as e:
                print(e)
                print(f'{i} SLEEP')
                sleep(60)


    def sample_openai_call(self,
                           prompt: str,
                           num_return_sequences: int,
                           temperature: float, top_p: float,
                           max_gen_length: int):
        for i in range(100):
            try:
                sleep(3)
                return self.sample_call_with_len(prompt=prompt,
                                                 num_return_sequences=num_return_sequences,
                                                 temperature=temperature, top_p=top_p,
                                                 max_gen_length=max_gen_length)
            except Exception as e:
                if 'maximum context length' in str(e):
                    max_gen_length -= 32
                    print(f'{i} REDUCING max_gen_length TO {max_gen_length}')
                else:
                    raise e


    def sample_call_with_len(
            self,
            prompt,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.95,
            max_gen_length=128,
            return_raw=False
    ):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            n=num_return_sequences,
            temperature=temperature,
            max_tokens=max_gen_length,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )

        if return_raw:
            return response
        result = []
        for completion in response['choices']:
            result.append(completion["message"]["content"])
        return result


api_key = ""
if __name__ == '__main__':
    # pytorch = OpenAIModel(model_name="text-davinci-003", api_key=api_key)
    model = OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key)
    # prompt = "Among the 7 words below (delimited by |), which one is the most semantically similar to the phrase \"double star\"? binary | double | star | dual | lumen | neutralism | keratoplasty. Please only tell me the word without any explanation "
    prompt = "Given two phrases \"access information\" and \"information access\", predict their semantic relatedness within the range [0, 1]"
    result = model.sample_call_with_len(prompt=prompt)
    print(result)
