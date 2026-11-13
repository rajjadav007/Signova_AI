# Task 2.2A.1: Signer 7 Validation Classification Analysis

**Model:** `best_dtw_knn_task22a1_model.pkl`  
**Split:** Signer 7 Validation Set (Strictly Zero Test Set Contamination)  
**Verification:** Total = 15 (Correct) + 70 (Incorrect) = 85 / 85 samples  
**Top-1 Validation Accuracy:** 17.65% (15/85)  

---

## 1. Summary of Predictions

| Metric | Value | Verification Status |
| :--- | :--- | :--- |
| **Total Validation Samples** | `85` | Confirmed (All Signer 7 samples) |
| **Correctly Classified Videos (Top-1)** | `15` | Confirmed (17.65%) |
| **Incorrectly Classified Videos** | `70` | Confirmed (82.35%) |
| **Count Sum Check** | `15 + 70 = 85` | Verified (`85 == 85`) |
| **Model Checkpoint** | `best_dtw_knn_task22a1_model.pkl` | Preserved (No retraining) |
| **Test Set Quarantine** | Signer 2 (98 samples) | Untouched (Zero access) |

---

## 2. Correctly Classified Videos (15 Samples)

These 15 validation samples from Signer 7 were correctly classified by the champion velocity-dominant DTW + KNN classifier (`win10_velo_dom_k9`):

| # | Video Filename | Video ID | Ground-Truth Class ID | Predicted Class ID | Sentence Text |
| :-: | :--- | :--- | :-: | :-: | :--- |
| 1 | `free (5).MP4` | `s000_signer7_free__5_` | `0` | `0` | **are you free today** |
| 2 | `worry (4).MP4` | `s014_signer7_worry__4_` | `14` | `14` | **do not worry** |
| 3 | `room (4).MP4` | `s019_signer7_room__4_` | `19` | `19` | **He is going into the room** |
| 4 | `today (5).MP4` | `s022_signer7_today__5_` | `22` | `22` | **he would be coming today** |
| 5 | `i am (age) (1).MP4` | `s030_signer7_i_am__age___1_` | `30` | `30` | **i am (age)** |
| 6 | `cold (5).MP4` | `s034_signer7_cold__5_` | `34` | `34` | **i am feeling cold** |
| 7 | `sitting (5).MP4` | `s040_signer7_sitting__5_` | `40` | `40` | **i am sitting in the class** |
| 8 | `MVI_6109.MP4` | `s045_signer7_MVI_6109` | `45` | `45` | **i can not help you there** |
| 9 | `i really appreciate it (3).MP4` | `s055_signer7_i_really_appreciate_it__3_` | `55` | `55` | **i really appreciate it** |
| 10 | `my name is xxxxxxxx (1).MP4` | `s061_signer7_my_name_is_xxxxxxxx__1_` | `61` | `61` | **my name is xxxxxxxx** |
| 11 | `pour (5).MP4` | `s065_signer7_pour__5_` | `65` | `65` | **pour some more water into the glass** |
| 12 | `take care (5).MP4` | `s070_signer7_take_care__5_` | `70` | `70` | **take care of yourself** |
| 13 | `MVI_6106.MP4` | `s073_signer7_MVI_6106` | `73` | `73` | **that is so kind of you** |
| 14 | `MVI_6208.MP4` | `s077_signer7_MVI_6208` | `77` | `77` | **we are all with you** |
| 15 | `leave (5).MP4` | `s088_signer7_leave__5_` | `88` | `88` | **when will the train leave** |

---

## 3. Incorrectly Classified Videos (70 Samples)

Detailed error breakdown for all 70 misclassified validation samples:

| # | Video Filename | Actual Class ID | Actual Sentence | Predicted Class ID | Predicted Sentence |
| :-: | :--- | :-: | :--- | :-: | :--- |
| 1 | `MVI_6209.MP4` | `1` | are you hiding something | `28` | *how dare you* |
| 2 | `bring water for me (8).MP4` | `2` | bring water for me | `87` | *what you want* |
| 3 | `help (5).MP4` | `3` | can i help you | `26` | *how can i help you* |
| 4 | `can you repeat that please (1).MP4` | `4` | can you repeat that please | `52` | *i like you i love you* |
| 5 | `comb (5).MP4` | `5` | comb your hair | `16` | *go and sleep* |
| 6 | `congratulations (5).MP4` | `6` | congratulations | `29` | *how old are you* |
| 7 | `could you please talk slower (3).MP4` | `7` | could you please talk slower | `52` | *i like you i love you* |
| 8 | `favour (5).MP4` | `8` | do me a favour | `26` | *how can i help you* |
| 9 | `abuse (4).MP4` | `9` | do not abuse him | `59` | *it was nice chatting with you* |
| 10 | `heart (5).MP4` | `13` | do not take it to the heart | `94` | *why are you disappointed* |
| 11 | `something (3).MP4` | `15` | do you need something | `39` | *i am really grateful* |
| 12 | `food (4).MP4` | `17` | had your food | `34` | *i am feeling cold* |
| 13 | `train (5).MP4` | `18` | he came by train | `83` | *what do you want to become* |
| 14 | `way (4).MP4` | `20` | he is on the way | `81` | *what do you do* |
| 15 | `friend (4).MP4` | `21` | he she is my friend | `16` | *go and sleep* |
| 16 | `help me (8).MP4` | `23` | help me | `28` | *how dare you* |
| 17 | `hi how are you (3).MP4` | `24` | hi how are you | `78` | *wear the shirt* |
| 18 | `things (4).MP4` | `25` | how are things | `40` | *i am sitting in the class* |
| 19 | `how can i help you (3).MP4` | `26` | how can i help you | `62` | *nice to meet you* |
| 20 | `trust (5).MP4` | `27` | how can i trust you | `74` | *This place is beautiful* |
| 21 | `how old are you (3).MP4` | `29` | how old are you | `94` | *why are you disappointed* |
| 22 | `afraid (5).MP4` | `31` | i am afraid of that | `12` | *do not make me angry* |
| 23 | `i am crying (5).MP4` | `32` | i am crying | `34` | *i am feeling cold* |
| 24 | `bored (5).MP4` | `33` | i am feeling bored | `42` | *i am suffering from fever* |
| 25 | `fine thank (5).MP4` | `35` | i am fine. thank you sir | `28` | *how dare you* |
| 26 | `i am hungry   (3).MP4` | `36` | i am hungry | `94` | *why are you disappointed* |
| 27 | `dilemma (5).MP4` | `37` | i am in dilemma what to do | `31` | *i am afraid of that* |
| 28 | `really sure (4).MP4` | `38` | i am not really sure | `42` | *i am suffering from fever* |
| 29 | `MVI_6104.MP4` | `39` | i am really grateful | `59` | *it was nice chatting with you* |
| 30 | `sorry hear (5).MP4` | `41` | i am so sorry to hear that | `52` | *i like you i love you* |
| 31 | `tired (5).MP4` | `43` | i am tired | `59` | *it was nice chatting with you* |
| 32 | `very happy (4).MP4` | `44` | i am very happy | `45` | *i can not help you there* |
| 33 | `agree (5).MP4` | `46` | i do not agree | `62` | *nice to meet you* |
| 34 | `hate (4).MP4` | `47` | i do not like it | `52` | *i like you i love you* |
| 35 | `mean (5).MP4` | `48` | i do not mean it | `59` | *it was nice chatting with you* |
| 36 | `enjoyed (4).MP4` | `50` | i enjoyed a lot | `72` | *thank you so much* |
| 37 | `got hurt (5).MP4` | `51` | i got hurt | `11` | *do not hurt me* |
| 38 | `MVI_6207.MP4` | `52` | i like you i love you | `30` | *i am (age)* |
| 39 | `i need water 7.MP4` | `53` | i need water | `20` | *he is on the way* |
| 40 | `promise (5).MP4` | `54` | i promise | `14` | *do not worry* |
| 41 | `somehow (5).MP4` | `56` | i somehow got to know about it | `94` | *why are you disappointed* |
| 42 | `stopped (3).MP4` | `57` | i was stopped by some one | `28` | *how dare you* |
| 43 | `difference (5).MP4` | `58` | it does not make any difference to me | `28` | *how dare you* |
| 44 | `never hurt (5).MP4` | `64` | now onwards he will never hurt you | `99` | *you do anything, i do not care* |
| 45 | `bed (5).MP4` | `66` | prepare the bed | `52` | *i like you i love you* |
| 46 | `serve (5).MP4` | `67` | serve the food | `52` | *i like you i love you* |
| 47 | `outside (4).MP4` | `68` | shall we go outside | `62` | *nice to meet you* |
| 48 | `softly (5).MP4` | `69` | speak softly | `96` | *you are good* |
| 49 | `truth (4).MP4` | `71` | tell me truth | `72` | *thank you so much* |
| 50 | `thank you so much (1).MP4` | `72` | thank you so much | `52` | *i like you i love you* |
| 51 | `understand (5).MP4` | `75` | try to understand | `47` | *i do not like it* |
| 52 | `turn on (5).MP4` | `76` | turn on light turn off light | `34` | *i am feeling cold* |
| 53 | `shirt (5).MP4` | `78` | wear the shirt | `16` | *go and sleep* |
| 54 | `what are you doing (1).MP4` | `79` | what are you doing | `23` | *help me* |
| 55 | `tell him (4).MP4` | `80` | what did you tell him | `84` | *what happened* |
| 56 | `what do you think (1).MP4` | `82` | what do you think | `32` | *i am crying* |
| 57 | `become (5).MP4` | `83` | what do you want to become | `52` | *i like you i love you* |
| 58 | `happened (3).MP4` | `84` | what happened | `52` | *i like you i love you* |
| 59 | `what is your phone number (1).MP4` | `86` | what is your phone number | `59` | *it was nice chatting with you* |
| 60 | `what you want (1).MP4` | `87` | what you want | `52` | *i like you i love you* |
| 61 | `college school (3).MP4` | `90` | which college school are you from | `62` | *nice to meet you* |
| 62 | `who are you (1).MP4` | `91` | who are you | `94` | *why are you disappointed* |
| 63 | `angry (5).MP4` | `92` | why are you angry | `42` | *i am suffering from fever* |
| 64 | `crying (5).MP4` | `93` | why are you crying | `42` | *i am suffering from fever* |
| 65 | `bad (5).MP4` | `95` | you are bad | `80` | *what did you tell him* |
| 66 | `good (5).MP4` | `96` | you are good | `23` | *help me* |
| 67 | `welcome (5).MP4` | `97` | you are welcome | `28` | *how dare you* |
| 68 | `can do (5).MP4` | `98` | you can do it | `16` | *go and sleep* |
| 69 | `dont care (2).MP4` | `99` | you do anything, i do not care | `59` | *it was nice chatting with you* |
| 70 | `medicine (5).MP4` | `100` | you need a medicine, take this one | `46` | *i do not agree* |

---

## 4. Integrity and Compliance Checklist

- [x] **No Model Retraining:** Evaluation executed strictly using pre-trained checkpoint `best_dtw_knn_task22a1_model.pkl`.
- [x] **No Checkpoint Modification:** Checkpoint timestamps and hashes remained unaltered.
- [x] **Test Set Untouched:** Only Signer 7 validation sequences were evaluated. Signer 2 test sequences were strictly quarantined.
- [x] **Deterministic Exact Counts:** Correct (15) + Incorrect (70) = Total (85) exactly.
- [x] **Exact Match to Report:** Top-1 validation accuracy matches Task 2.2A.1 report `17.65% (15/85)`.
