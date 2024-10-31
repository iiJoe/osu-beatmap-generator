from pydub import AudioSegment
import os
import constants
import glob
import numpy as np

# Split the osu files into the following sections
# ['General', 'Editor', 'Metadata', 'Difficulty', 'Events', 'TimingPoints', 'Colours', 'HitObjects']
def parse_sections(file_path):
    sections = {}
    current_section = None
    current_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for section label e.g. ["General"]
            if line.startswith('[') and line.endswith(']'):
                if current_section:
                    sections[current_section] = current_lines

                current_section = line[1:-1]  # Remove square brackets
                current_lines = []

            elif line == '':
                if current_section:
                    sections[current_section] = current_lines
                    current_section = None
                    current_lines = []

            elif current_section:
                current_lines.append(line)

        if current_section:
            sections[current_section] = current_lines

    return sections

# Splits a "Key: Value" string into kv pairs
def data_from_section(section):
  kv_pairs = {}

  for line in section:
      key, value = line.split(':', 1)
      kv_pairs[key.strip()] = value.strip()

  return kv_pairs

# See https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29#hit-objects
def hit_objects_dict(list):
  hit_objects = []
  for item in list:
    values = item.split(",")
    hit_object = dict(zip(constants.osu_keys, values))
    hit_objects.append(hit_object)

  return hit_objects

# 10.24s intervals
def splice_audio(file_path, beatmap_id, interval_ms=10240):
    _, file_name = os.path.split(file_path)
    prefix, file_extension = os.path.splitext(file_name)
    file_extension = file_extension[1:] # Remove leading dot
    new_directory = constants.splice_directory

    if os.path.exists(new_directory) == False:
      os.mkdir(new_directory)

    audio = AudioSegment.from_file(file_path)
    audio_duration = len(audio)
    audio_splices = []

    for i in range(0, audio_duration, interval_ms):

        chunk = audio[i:i + interval_ms]
        chunk_name = os.path.join(new_directory, f"{beatmap_id}-{prefix}_{i // interval_ms}.{file_extension}")
        chunk.export(chunk_name, format=file_extension)
        audio_splices.append(chunk_name)

    return audio_splices

def preprocess():
    osu_folders = [name for name in os.listdir(constants.beatmaps_directory) if os.path.isdir(os.path.join(constants.beatmaps_directory, name))]

    # Clearing the labels file to ensure no duplicates of data
    with open(constants.training_labels_file, 'w') as file:
        pass
    with open(constants.test_labels_file, 'w') as file:
        pass

    beatmaps_count = 0
    splices_count = 0
    total_rows = []
    for beatmap_set_id in osu_folders:

        new_directory = constants.beatmaps_directory + '/' + beatmap_set_id
        osu_files = glob.glob(os.path.join(new_directory, '*.osu'))

        for new_file in osu_files:
          beatmaps_count += 1
          file_sections = parse_sections(new_file)
          general_data = data_from_section(file_sections["General"])
          editor_data = data_from_section(file_sections["Editor"])
          meta_data = data_from_section(file_sections["Metadata"])
          difficulty_data = data_from_section(file_sections["Difficulty"])
          hit_objects_data = hit_objects_dict(file_sections["HitObjects"])

          beatmap_id = meta_data["BeatmapID"]

          # Splicing audio into intervals
          audio_file = general_data["AudioFilename"]
          audio_directory = new_directory + "/" + audio_file
          spliced_audio_paths = splice_audio(audio_directory, beatmap_id)
          splices_count += len(spliced_audio_paths)

          # Beat timings
          hit_timings_data = [int(hit_object["time"]) for hit_object in hit_objects_data]
          hit_timings = np.full((len(spliced_audio_paths), 1024), "0", dtype=str)
          for timing in hit_timings_data:
            hit_timings[timing // 10240][(timing % 10240) // 10] = "1"
          prepended_paths = np.array(spliced_audio_paths).reshape(-1, 1)
          rows = np.concatenate((prepended_paths, hit_timings), axis=1)
          total_rows.append(rows)

    total_rows = np.vstack(total_rows)
    shape = total_rows.shape[0]
    train_rows, test_rows = np.split(total_rows[np.random.permutation(shape)], [int(0.8 * shape)])

    with open(constants.training_labels_file, "ab") as file:
        np.savetxt(file, train_rows, delimiter=",", fmt="%s")
    with open(constants.test_labels_file, "ab") as file:
        np.savetxt(file, test_rows, delimiter=",", fmt="%s")

    print(f"Processed {beatmaps_count} beatmaps with {splices_count} splices of audio")
    print(f"Split dataset into {train_rows.shape[0]} splices for training and {test_rows.shape[0]} splices for test")

preprocess()
