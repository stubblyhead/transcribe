from faster_whisper import WhisperModel
import ffmpeg
from multiprocessing import cpu_count, Process, Pipe
from typing import List
import concurrent.futures
import time
import os
from split import split_audio_into_chunks


def transcribe_file(file_path, model, conn):
    print(f'transcribing {os.path.basename(file_path)} from parallel process')
    segments, info = model.transcribe(file_path)
    print(f'finished transcribing {os.path.basename(file_path)}, a {info.duration} sec long audio file')
    segments = list(segments)
    conn.send([segments])
    conn.close


def transcribe_audio(input_file: str, max_processes = 0,
                     silence_threshold: str = "-20dB", silence_duration: float = 2.0, model=None) -> str:
    if max_processes > cpu_count() or max_processes == 0:
        max_processes = cpu_count()

    # Split the audio into chunks
    print('splitting audio into chunks')
    temp_files_array = split_audio_into_chunks(input_file, max_processes, silence_threshold, silence_duration)
    start = time.time()
    futures = []
    processes = []
    parent_connections = []
#    print(f'starting {max_processes} concurrent transcription process')
    for file_path in temp_files_array:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        process = Process(target=transcribe_file,args=(file_path, model, child_conn))
        processes.append(process)
    for process in processes:
        print(f'starting process for {os.path.basename(file_path)}')
        process.start()

    for process in processes:
        process.join()

    result_string = ''
    for parent_connection in parent_connections:
        result_string += parent_connection.recv()[0]

    # Submit each file to the thread pool and store the corresponding future object
#    with concurrent.futures.ThreadPoolExecutor(max_processes) as executor:
#        for file_path in temp_files_array:
#            future = executor.submit(transcribe_file, file_path, model)
#            futures.append(future)

#   result_string = ""
#   for future in futures:
#        segments = future.result()
#        for seg in segments:
        #    result_string += "".join(segment.text for segment in segments)
#            result_string += f"{seg.text}\n"

    end = time.time()
    print(end - start)

    # Remember to remove the temporary files after you're done processing them
    for temp_file in temp_files_array:
        os.remove(temp_file)
    return result_string


