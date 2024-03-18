import os
import json
import urllib.parse
from faster_whisper import WhisperModel
import boto3
import time
import split
import transcribe

dest_bucket = os.environ.get('DEST_BUCKET')
silence_duration = float(os.environ.get('SILENCE_DURATION'))
silence_threshold = os.environ.get('SILENCE_THRESH')
s3 = boto3.client("s3")

def handler(event, context):
    print('## EVENT')
    print(event)
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        print("Bucket:", bucket, "key:", key)
        prefix = os.path.dirname(key)
        filename = os.path.basename(key)
        audio_file = f'/tmp/{filename}'
        print(f"saving object to {audio_file}.")
        # Downloading file to transcribe
        s3.download_file(bucket, key, audio_file)
        device = "cpu"
        model_size = 'medium.en'
        start_time = time.time()
        model = WhisperModel(model_size, device="cpu", compute_type="auto", download_root='/usr/local', local_files_only=True, cpu_threads = 6)
        #turn on debug logging
        model.logger.setLevel(10)
        print(f"loaded {model_size} model in {time.time()-start_time:.02f} seconds, starting file {filename}")

        start_time = time.time()
#        segments, info = model.transcribe(audio_file, beam_size=5, language='en')
#        output = ''
#        for segment in segments:
#            output += ("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

        max_processes = 6 
        output = transcribe.transcribe_audio(audio_file, max_processes, silence_threshold=silence_threshold, silence_duration=silence_duration, model=model)
        object = s3.put_object(Bucket=dest_bucket, Key=f'{key}.text', Body=output)
        os.remove(audio_file)
        print(f"finished {filename} in {time.time()-start_time:.02f} sec, wrote output to s3://{dest_bucket}/{key}.text")
        try:
            # Generate a pre-signed URL for the S3 object
            expiration = 3600  # URL expiration time in seconds
            response = s3.generate_presigned_url('get_object',
                                                    Params={'Bucket': dest_bucket, 'Key': f'text/{filename}.text'},
                                                    ExpiresIn=expiration)

            output = f"Transcribed: {filename}.text - {response}"

        except ClientError as e:
            print(e)

        return {
            "statusCode": 200,
            "body": json.dumps(output)
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps("Error processing the file")
        }
