import json
import torch
import torchaudio
import IPython
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import phonemizer
from phonemizer.separator import Separator

@dataclass
class Segment:
    id: int
    label: str
    start: float
    end: float
    score: float

    def __repr__(self):
        return (
            f"{self.label}\t{self.id}\t({self.score:3f}):\t[{self.start},\t{self.end})"
        )

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float



@dataclass
class PhoneSegment:
    id:int
    label:str
    start:float
    end:float
    lpp:list

class GOP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained("model")
    processor = Wav2Vec2Processor.from_pretrained("model")
    emission = None
    
    def forward(self, audio_path, transcript):
        print(f'transcript is: {transcript}')
        waveform, sample_rate = torchaudio.load(audio_path)
        print('audio loaded! sample rate is:', sample_rate)
        if sample_rate != self.processor.feature_extractor.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.processor.feature_extractor.sampling_rate
            )(waveform)
            sample_rate = self.processor.feature_extractor.sampling_rate

        audio_input_values = self.processor(waveform.squeeze(), return_tensors="pt").input_values
        self.audio_duration_sec = audio_input_values.shape[1] / sample_rate

        print(f'audio_duration_sec: {self.audio_duration_sec}')
        with torch.inference_mode():
            outputs = self.model(audio_input_values)

        self.emission = outputs.logits[0]
        logits = outputs.logits[torch.argmax(outputs.logits, dim=-1) != 0]
        logits = torch.softmax(logits, dim=-1)
        
        predicted_phones, real_phones, word_pos = self.get_transcription(transcript, logits)
        print(f'predicted phones: {predicted_phones}')
        print(f'real phones: {real_phones}')
        print(f'word_pos: {word_pos}')
        aligned_segments, aligned_predicted_segments = self.align_phones(real_phones, predicted_phones)
        
        scores = self.gen_scores(aligned_segments, aligned_predicted_segments, logits, transcript, word_pos, real_phones)
        
        return scores
        
        
    def get_transcription(self, transcript, logits):
        predicted_phones = [
            self.processor.tokenizer.decoder[id.item()]
            for id in torch.argmax(logits, -1)
        ]
        
        separator = Separator(phone="-", word="|")
        real_phones = []
        word_pos = []
        i = 0
        word_i = 0
        words_in_the_transcript = transcript.split()

        for word in words_in_the_transcript:
            new_word_phones = phonemizer.phonemize(word, strip=True, separator=separator).split("-")
            word_pos.append([i, i + len(new_word_phones)])
            real_phones += new_word_phones
            i += len(new_word_phones)
            
        return predicted_phones, real_phones, word_pos
    
    def align_phones(self, real_phones, predicted_phones):
        def get_trellis(emission, tokens, blank_id=0):
            num_frame = emission.size(0)
            num_tokens = len(tokens)

            trellis = torch.zeros((num_frame, num_tokens))
            trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
            trellis[0, 1:] = -float("inf")
            trellis[-num_tokens + 1 :, 0] = float("inf")

            for t in range(num_frame - 1):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens[1:]],
                )
            return trellis


        def merge_repeats(path, transcript, trellis, audio_duration_sec):
            ratio = audio_duration_sec / trellis.size(0)
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                if transcript[path[i1].token_index] != self.processor.tokenizer.pad_token:
                    segments.append(
                        Segment(
                            self.processor.tokenizer.convert_tokens_to_ids(
                                transcript[path[i1].token_index]
                            ),
                            transcript[path[i1].token_index],
                            path[i1].time_index * ratio,
                            (path[i2 - 1].time_index + 1) * ratio,
                            score,
                        )
                    )
                i1 = i2
            return segments
        
        
        def backtrack(trellis, emission, tokens, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1

            path = [Point(j, t, emission[t, blank_id].exp().item())]
            while j > 0:
                # Should not happen but just in case
                assert t > 0

                # 1. Figure out if the current position was stay or change
                # Frame-wise score of stay vs change
                p_stay = emission[t - 1, blank_id]
                p_change = emission[t - 1, tokens[j]]

                # Context-aware score for stay vs change
                stayed = trellis[t - 1, j] + p_stay
                changed = trellis[t - 1, j - 1] + p_change

                # Update position
                t -= 1
                if changed > stayed:
                    j -= 1

                # Store the path with frame-wise probability.
                prob = (p_change if changed > stayed else p_stay).exp().item()
                path.append(Point(j, t, prob))

            # Now j == 0, which means, it reached the SoS.
            # Fill up the rest for the sake of visualization
            while t > 0:
                prob = emission[t - 1, blank_id].exp().item()
                path.append(Point(j, t - 1, prob))
                t -= 1

            return path[::-1]


        def align(text):
            transcript = text
            indexed_tokens = [self.processor.tokenizer.encoder.get(c, self.processor.tokenizer.pad_token_id) for c in transcript]

            trellis = get_trellis(self.emission, indexed_tokens)
            path = backtrack(trellis, self.emission, indexed_tokens)

            aligned_segments = merge_repeats(path, transcript, trellis, self.audio_duration_sec)

            return aligned_segments

        aligned_segments = align(real_phones)
        aligned_predicted_segments = align(predicted_phones)

        print(aligned_segments)
        print(aligned_predicted_segments)
        
        return aligned_segments, aligned_predicted_segments
        
    def gen_scores(self, aligned_segments, aligned_predicted_segments, logits, transcript, word_pos, real_phones):
        merged_gt = []
        merged_p = []
        score = []
        i = 0
        duration = 0

        final_feat = []

        logit_id = 0


        for id, gt_frame in enumerate(aligned_segments):
            if i>=1:
                i-=1
            while i<logits.shape[0] and aligned_predicted_segments[i].end < gt_frame.start: i += 1
            duration = 0
            tmp_lpp_part = torch.zeros(logits.shape[1])
            while i<logits.shape[0] and aligned_predicted_segments[i].start < gt_frame.end:
                duration += 1
                tmp_lpp_part += logits[i] * (
                    (min(gt_frame.end, aligned_predicted_segments[i].end) - max(gt_frame.start, aligned_predicted_segments[i].start))
                    /
                    (max(gt_frame.end, aligned_predicted_segments[i].end) - min(gt_frame.start, aligned_predicted_segments[i].start))
                )
                i+=1
            if duration:
                tmp_lpp_part /= duration
            gop = tmp_lpp_part[gt_frame.id] - max(tmp_lpp_part)
            score.append([self.processor.tokenizer.decoder[gt_frame.id], self.processor.tokenizer.decoder[np.argmax(tmp_lpp_part).item()], 100*(10**float(gop))])
        # score

        words = transcript.split()
        final_result = []
        for word, w_pos in zip(words, word_pos):
            final_result.append({
                'word': word,
                'phones': [],
            })
            print(word, '->', real_phones[w_pos[0]: w_pos[1]])
            for i in range(w_pos[0], w_pos[1]):
                final_result[-1]['phones'].append({
                    'real_phone': score[i][0],
                    'predicted_phone': score[i][1],
                    'score': f'{score[i][2]:.2f}'
                })
                print('', score[i], sep='\t')
            print()
        return final_result

