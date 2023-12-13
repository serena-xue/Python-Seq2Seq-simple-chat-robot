from parameters import *


def process_dialog(input_file_name, output_file_name):
    with open(root_path + input_file_name, 'r', encoding='utf-8') as f:
        with open(root_path + output_file_name, 'w', encoding='utf-8') as f_write:
            for dialogue in f.readlines():
                sentences = dialogue.strip('\n').strip(' __eou__').split(' __eou__')
                # Remove the last sentence of odd-numbered dialogues
                if len(sentences) % 2 == 1:
                    sentences.pop()
                flag = 1
                for sentence in sentences:
                    sentence = sentence.strip(' ').lower()
                    # Set the dialogue: Speaker1\tSpeaker2
                    if flag % 2 == 1:
                        f_write.write(sentence + '\t')
                    else:
                        f_write.write(sentence + '\n')
                    flag += 1
            f_write.close()
        f.close()
    print('done.')


process_dialog(input_file_name='dialogues_validation.txt', output_file_name='dialogues_validation_processed.txt')
