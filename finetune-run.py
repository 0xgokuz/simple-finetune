import json
from time import sleep
from openai import OpenAI

client = OpenAI()

def wait_untill_done(job_id): 
    events = {}
    while True: 
        response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
        # collect all events
        for event in response.data:
            if "data" in event and event.data:
                events[event.data.step] = event.data.train_loss
        messages = [it.message for it in response.data]
        for m in messages:
            if m.startswith("New fine-tuned model created: "):
                return m.split("created: ")[1], events
        sleep(10)

if __name__ == "__main__":
    response = client.files.create(file=open("data/train.jsonl", "rb"), purpose="fine-tune")
    uploaded_id = response.id
    print("Dataset is uploaded")
    print("Sleep 30 seconds...")
    sleep(30)  # wait until dataset would be prepared
    response = client.fine_tuning.jobs.create(training_file=uploaded_id,model="gpt-3.5-turbo",hyperparameters={"n_epochs": 10})
    print("Fine-tune job is started")
    ft_job_id = response.id
    new_model_name, events = wait_untill_done(ft_job_id)
    with open("result/new_model_name.txt", "w") as fp:
        fp.write(new_model_name)
    print(new_model_name)