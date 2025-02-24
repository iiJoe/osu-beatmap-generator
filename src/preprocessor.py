import os
import json
import constants
import glob
from utils import splice_audio

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

def preprocess():
    osu_folders = [name for name in os.listdir(constants.beatmaps_directory) if os.path.isdir(os.path.join(constants.beatmaps_directory, name))]

    if not osu_folders:
        print(f"No beatmap folders found in {constants.beatmaps_directory}.")
        return

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
