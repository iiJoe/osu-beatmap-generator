from pydub import AudioSegment
import os
import json
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
    hit_object = dict(zip(constants.hit_object_keys, values))
    hit_objects.append(hit_object)

  return hit_objects

# 10.24s intervals
def splice_audio(file_path, beatmap_id, interval_ms=constants.seq_length * 10):
    _, file_name = os.path.split(file_path)
    prefix, file_extension = os.path.splitext(file_name)
    file_extension = file_extension[1:] # Remove leading dot
    new_directory = constants.splice_directory

    if not os.path.exists(new_directory):
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

    beatmaps_count = 0
    splices_count = 0
    total_rows = []
    if not osu_folders:
        print("No beatmap folders found.")

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

          # Only considering hit circle notes, may consider spinners and sliders in the future
          hit_timing_delimiter = constants.seq_length * 10
          note_index = 0
          type_bitmask = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) # Only considering hit circle and combo bits
          for i in range(0, len(spliced_audio_paths)):
              if note_index == len(hit_objects_data):
                  break

              attributes = []
              hit_object = hit_objects_data[note_index]

              while (int(hit_object["time"]) < (i + 1) * hit_timing_delimiter):
                  tpe = (int(hit_object["type"]) & type_bitmask)
                  if not tpe:
                      tpe = 1

                  attributes.append({
                      "x" : int(hit_object["x"]),
                      "y" : int(hit_object["y"]),
                      "time": int(hit_object["time"]) % hit_timing_delimiter // 10,
                      "type": tpe,
                      "hitSound": int(hit_object["hitSound"])
                  })

                  note_index += 1
                  if note_index == len(hit_objects_data):
                      break

                  hit_object = hit_objects_data[note_index]

              data = {
                  constants.json_file_path_key: spliced_audio_paths[i],
                  constants.json_attributes_key: attributes
              }
              total_rows.append(data)

    if total_rows:
        split_delimiter = int(len(total_rows) * 0.8)
        train_rows, test_rows = [total_rows[:split_delimiter], total_rows[split_delimiter:]]
        with open(constants.training_labels_file, "w") as file:
            json.dump(train_rows, file, indent=4)
        with open(constants.test_labels_file, "w") as file:
            json.dump(test_rows, file, indent=4)

        print(f"Processed {beatmaps_count} beatmaps with {splices_count} splices of audio")
        print(f"Split dataset into {len(train_rows)} splices for training and {len(test_rows)} splices for test")

preprocess()
