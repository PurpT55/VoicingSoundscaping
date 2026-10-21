# Emotions

VSS speaks one emotion vocabulary for every backend: **96 entries** in four kinds, plus **23 nonverbal sounds**.

* **emotion**: an affective state (joy, grief, fear ...)
* **attitude**: a social colour (dry humor, cheeky, sarcastic, comforting ...)
* **state**: a physical condition that changes the voice (crying, sleepy, out of breath ...)
* **delivery**: a vocal mode that combines with any of the above (whisper, shout, scream ...)

```
MIRA (whisper, afraid 0.7, "afraid to wake something"): Is anyone up there?
TOM (crying 0.4, tender 0.5): Every single night. <sniff> [grateful 0.6] But tonight the light came back on.
```

* Intensity runs 0..1 and defaults to 0.6 ("clearly present"). Below 0.3 is subtle, above 0.9 extreme.
* Several parts blend: `(sad 0.6, relief 0.4)`. A free-text acting note goes in quotes.
* `[...]` switches emotion mid-line, relative to the line's own emotion. `[reset]` goes back.
* The cast `default:` (e.g. `TOM: {default: dry_humor 0.3}`) applies under every line. A line's
  own emotion replaces it, but a default *delivery* persists.

## How each backend performs it (`src/vss/emotion/mapping.py`)

| mechanism | backends | what VSS sends |
|---|---|---|
| 8-dim emotion vector | IndexTTS-2/2.5 | `emo_vector` scaled by intensity, sum capped at 0.8 |
| emotion reference clip | IndexTTS (any speaker), Chatterbox/Qwen3-clone (same voice) | `emo_audio_prompt` from the emotion bank, `emo_alpha` from intensity |
| free-form inline tags | Fish S2-Pro / Drama 3, Higgs-style servers | `[whispering] [very afraid] [afraid to wake something] text` |
| natural-language instruct | Qwen3-TTS, OpenAI-style servers | "Afraid and frightened ... (strong intensity). Delivery: Whispering ..." |
| edit passes | Step-Audio-EditX | `emotion:fear` / `style:whisper`, 1-3 iterations by intensity |
| exaggeration | Chatterbox | `exaggeration` / `cfg_weight` from arousal and intensity |
| prosody only | Kokoro, espeak | speed/pitch; the mixer keeps whispers quiet and shouts big |

Every entry also carries prosody hints (`rate`, `pitch`, `level`). They are used by speed controls
(IndexTTS-2.5 `duration_factor`, Kokoro `speed`), by DSP fallbacks, and by the mixer. Whispers sit
about 8 dB under normal speech and shouts about 5 dB over it, so dynamics survive loudness
normalisation.

## Coverage: the 80-90 % goal

A coverage level is computed per emotion:
* **native**: a direct control exists.
* **bank**: native once a reference clip exists.
* **approx**: only a neighbouring emotion or prosody is reachable.
* **none**: no control.

This measures *controllability*, not perceived quality. Listening tests (`QA_SHEET.md`) and
automatic scoring (ASR + emotion2vec) are the quality side. Regenerate the tables with `vss emotions --markdown`.

Ways to reach the goal:
1. **Fish Drama 3** (48 GB VRAM): 100 % native, best at crying, screams and laughter.
2. **IndexTTS-2.5 + a full shared emotion bank**: 100 %, on a 12-16 GB GPU. Fill the bank once
   with `vss bank build --backend fish|qwen3`, or record people acting the lines.
3. **Fish S2-Pro or Qwen3-TTS**: 87.5 %. The `hard` extremes are approximated, so route them:
   `routing: {hard: fish}` with Drama 3, or bank clips.
4. **Step-Audio-EditX over IndexTTS**: refinement that also adds sighs, laughs and breaths.

Routing keys under `backends.routing`: an emotion id (`scream: fish`), `hard`, `kind:delivery`, `family:sadness`.

## Emotion banks

`banks/emotions/<id>.wav` (+ `.txt`), with numbered variants `<id>_2.wav`. Per-voice banks go in
`voices/<voice>/emotions/`. The lookup order is: delivery clip first (whisper, scream, sob are the
hardest to fake), then the primary emotion; voice bank first, then the shared bank (only for
backends that accept any speaker). `vss bank list -v` shows the gaps.

Every taxonomy entry ships a reference line written for that emotion (`vss emotions --list`). Matching
words to feeling gives the cleanest references.

## Reference tables (generated)

| id | kind | family | acting note | aliases | Step-Audio-EditX |
|---|---|---|---|---|---|
| `joy` | emotion | joy | Joyful and bright, smiling through every word. | joyful, happy, glad, happiness | emotion:happy |
| `delight` | emotion | joy | Delighted and charmed, a light lift of pleasant surprise. | delighted, charmed | emotion:happy |
| `excited` | emotion | joy | Excited and eager, words tumbling out quickly with bright energy. | eager, thrilled, enthusiastic, excitement | emotion:excited |
| `ecstatic` | emotion | joy | Ecstatic and overjoyed, bursting with uncontainable happiness. | elated, overjoyed, euphoric | emotion:excited |
| `amused` | emotion | joy | Amused and tickled, a chuckle sitting just under the words. | tickled, entertained, amusement | emotion:humour |
| `playful` | attitude | joy | Playful and light, bouncing rhythm, having fun. | fun, frolicsome | emotion:humour |
| `cheeky` | attitude | joy | Cheeky and impish, a grin you can hear, teasing without malice. | impish, mischievous, sassy, cheekily | emotion:humour |
| `teasing` | attitude | joy | Teasing and affectionate, drawing out the words with a knowing smile. | ribbing, joshing | emotion:humour |
| `giddy` | emotion | joy | Giddy and bubbly, barely holding back giggles. | giggly, bubbly | emotion:excited |
| `peppy` | attitude | joy | Peppy and perky, upbeat sparkle, energetic and chipper. | perky, chipper, upbeat, peaky, bouncy | style:vivid |
| `proud` | emotion | joy | Proud and warm, chest lifted, savouring the accomplishment. | pride, accomplished | emotion:happy |
| `triumphant` | emotion | joy | Triumphant and victorious, ringing out with power. | victorious, exultant | emotion:excited |
| `relieved` | emotion | joy | Relieved, tension melting away, a long breath out. | relief | emotion:happy |
| `content` | emotion | joy | Contented and cosy, unhurried, quietly happy. | contentment, satisfied, at_peace, cozy, cosy | style:warm |
| `grateful` | emotion | joy | Grateful and heartfelt, sincere thanks with a touch of emotion. | thankful, gratitude | style:soulful |
| `tender` | emotion | joy | Tender and loving, soft and close, full of affection. | loving, affectionate, love, fond | style:gentle |
| `hopeful` | emotion | joy | Hopeful and lifting, a gentle brightness reaching forward. | optimistic, hope | emotion:happy |
| `warm` | attitude | joy | Warm and friendly, welcoming, an open smile in the voice. | friendly, kind, welcoming | style:warm |
| `admiring` | emotion | joy | Admiring and impressed, warm respect and a little wonder. | admiration, impressed, respectful | emotion:admiration |
| `awe` | emotion | surprise | Awestruck and full of wonder, hushed and slow, eyes wide. | awestruck, wonder, amazed, marvelling, wonderstruck | emotion:admiration |
| `sad` | emotion | sadness | Sad and heavy, voice lowered, the words weighed down. | unhappy, down, sorrowful, sadness | emotion:sad |
| `melancholic` | emotion | sadness | Melancholic and grey, a slow quiet ache. | melancholy, gloomy, blue | emotion:sad |
| `wistful` | emotion | sadness | Wistful and nostalgic, a bittersweet smile at a faraway memory. | nostalgic, longing, bittersweet, yearning | style:soulful |
| `heartbroken` | emotion | sadness | Heartbroken and devastated, voice cracking with grief. | grief, grieving, devastated, anguished | emotion:sad |
| `crying` | state | sadness | Crying, speaking through tears, voice breaking and wet, catching breath between words. | cry, tearful, weeping, through_tears, teary | emotion:sad |
| `sobbing` | delivery | sadness | Sobbing, words broken by heaving sobs and ragged breaths. | sobs, bawling, wailing | emotion:sad |
| `despair` | emotion | sadness | Despairing and hopeless, the bottom has fallen out. | hopeless, despairing, desperate | emotion:depressed |
| `lonely` | emotion | sadness | Lonely and small, quiet and a little hollow. | alone, isolated | emotion:sad |
| `disappointed` | emotion | sadness | Disappointed and deflated, the hope draining out of the sentence. | let_down, deflated, disappointment | emotion:sad |
| `regretful` | emotion | sadness | Regretful and remorseful, heavy with what cannot be undone. | remorseful, rueful, regret | emotion:sad |
| `hurt` | emotion | sadness | Hurt and wounded, a sting of betrayal in the voice. | wounded, stung | emotion:sad |
| `depressed` | emotion | sadness | Dejected and flat, low energy, colour drained from the voice. | dejected, discouraged, downcast | emotion:depressed |
| `sympathetic` | emotion | sadness | Sympathetic and compassionate, gentle understanding. | empathetic, compassionate, understanding, empathy | emotion:empathy |
| `angry` | emotion | anger | Angry and sharp, hard consonants, heat in every word. | mad, anger, cross | emotion:angry |
| `furious` | emotion | anger | Furious and enraged, explosive, barely contained. | rage, enraged, livid, seething, wrathful | emotion:angry |
| `irritated` | emotion | anger | Irritated and annoyed, clipped and impatient. | annoyed, irked, bothered | emotion:angry |
| `frustrated` | emotion | anger | Frustrated and exasperated, at the very end of one's patience. | exasperated, fed_up, frustration | emotion:angry |
| `peeved` | emotion | anger | Peeved and huffy, mildly offended, pouting a little. | miffed, put_out, huffy, sulky | emotion:angry |
| `indignant` | emotion | anger | Indignant and affronted, righteous outrage. | outraged, affronted | emotion:angry |
| `bitter` | emotion | anger | Bitter and resentful, sour edges on the words. | resentful, sour | emotion:coldness |
| `contemptuous` | attitude | disgust | Contemptuous and scornful, looking down the nose. | scornful, disdainful, sneering, contempt | emotion:disgusted |
| `defiant` | attitude | anger | Defiant and unbowed, firm and unyielding. | rebellious, stubborn, unyielding | style:authority |
| `menacing` | attitude | anger | Quietly menacing, low and controlled, a threat beneath every word. | threatening, sinister, villainous, ominous | style:deeply |
| `grumpy` | emotion | anger | Grumpy and grouchy, gravelly grumbling. | cranky, grouchy, curmudgeonly | emotion:coldness |
| `jealous` | emotion | anger | Jealous and envious, a sulky edge. | envious, envy, jealousy | emotion:angry |
| `afraid` | emotion | fear | Afraid and frightened, voice tight and unsteady. | scared, fearful, frightened, fear | emotion:fear |
| `terrified` | emotion | fear | Terrified, raw panic, high and breathless. | horrified, petrified, terror | emotion:fear |
| `anxious` | emotion | fear | Anxious and nervous, jittery rhythm, slightly rushed. | nervous, uneasy, jittery, on_edge, anxiety | emotion:fear |
| `worried` | emotion | fear | Worried and concerned, a furrowed-brow tone. | concerned, troubled, worry | emotion:fear |
| `panicked` | emotion | fear | Panicked and frantic, tumbling words, gasping for air. | frantic, alarmed, panic | emotion:fear |
| `suspicious` | attitude | fear | Suspicious and wary, narrowed eyes, measured and probing. | wary, distrustful, skeptical, sceptical | emotion:coldness |
| `trembling` | delivery | fear | Voice trembling and unsteady, barely controlled. | shaky, quavering, voice_shaking | emotion:fear |
| `surprised` | emotion | surprise | Surprised, caught off guard, pitch jumping up. | surprise, startled | emotion:surprised |
| `shocked` | emotion | surprise | Shocked and stunned, disbelief knocking the wind out. | stunned, aghast, shock | emotion:surprised |
| `confused` | emotion | surprise | Confused and puzzled, searching, rising question marks. | puzzled, bewildered, perplexed, confusion | emotion:confusion |
| `curious` | emotion | surprise | Curious and intrigued, leaning in, a playful question in the voice. | intrigued, inquisitive, curiosity | emotion:happy |
| `disgusted` | emotion | disgust | Disgusted and grossed out, recoiling. | grossed_out, repulsed, disgust, revolted | emotion:disgusted |
| `neutral` | emotion | calm | Neutral and natural, clear and even. | plain, normal, default |  |
| `calm` | emotion | calm | Calm and steady, grounded, unhurried. | composed, steady | style:gentle |
| `serene` | emotion | calm | Serene and peaceful, soft and spacious. | peaceful, tranquil | style:ethereal |
| `storytelling` | attitude | calm | Engaged storytelling narration, warm, clear and well-paced, drawing the listener in. | narration, narrative, storyteller, narrator | style:story |
| `solemn` | attitude | calm | Solemn and serious, measured, carrying weight. | serious, grave, formal | style:serious |
| `mysterious` | attitude | calm | Mysterious and hushed, as if sharing an ancient secret. | enigmatic, cryptic, conspiratorial, secretive | style:murmur |
| `dreamy` | emotion | calm | Dreamy and faraway, drifting, soft focus. | faraway, daydreaming, wistful_dreamy | style:ethereal |
| `bored` | emotion | low_energy | Bored and unimpressed, flat and monotone, dragging the words. | uninterested, unimpressed, monotone, boredom | emotion:coldness |
| `indifferent` | attitude | low_energy | Indifferent and detached, cool and uninvested. | apathetic, detached, cold, aloof | emotion:coldness |
| `tired` | state | low_energy | Tired and weary, heavy and slow, low on breath. | weary, fatigued, worn_out | emotion:depressed |
| `exhausted` | state | low_energy | Exhausted and drained, every word an effort. | drained, spent | emotion:depressed |
| `sleepy` | state | low_energy | Sleepy and drowsy, words slow and slurred at the edges, fighting a yawn. | drowsy, yawning, half_asleep | style:murmur |
| `unwell` | state | low_energy | Unwell and congested, croaky and low on energy. | sick, ill, congested, poorly |  |
| `out_of_breath` | state | vocal | Out of breath, panting between phrases after running. | panting, breathless, winded |  |
| `dry_humor` | attitude | social | Bone-dry deadpan wit, understated and unhurried; the joke lives in what is not emphasised. | deadpan, dry, wry, droll, dry_humour | emotion:humour |
| `sarcastic` | attitude | social | Sarcastic and ironic, exaggerated sweetness with a sharp edge. | ironic, sardonic, mocking, sarcasm | emotion:humour |
| `smug` | attitude | social | Smug and self-satisfied, savouring being right. | self_satisfied, cocky, gloating | style:arrogant |
| `arrogant` | attitude | social | Arrogant and haughty, grand and condescending. | haughty, condescending, pompous | style:arrogant |
| `shy` | attitude | social | Shy and bashful, small voice, hesitant. | bashful, timid, meek | style:shy |
| `embarrassed` | emotion | social | Embarrassed and flustered, a nervous laugh at the edges. | flustered, sheepish, mortified | emotion:embarrass |
| `apologetic` | attitude | social | Apologetic and contrite, soft and sincere. | sorry, contrite | emotion:empathy |
| `pleading` | attitude | social | Pleading and imploring, urgent and vulnerable. | begging, imploring, beseeching | emotion:sad |
| `comforting` | attitude | social | Comforting and reassuring, soothing and close. | reassuring, soothing, consoling | style:comfort |
| `gentle` | attitude | social | Gentle and soft-spoken, kind and careful. | soft_spoken, kindly | style:gentle |
| `authoritative` | attitude | social | Authoritative and commanding, firm and clear. | commanding, stern, firm | style:authority |
| `determined` | attitude | social | Determined and resolute, steely focus. | resolute, steely, focused | style:authority |
| `sincere` | attitude | social | Sincere and earnest, open-hearted. | earnest, heartfelt, honest | style:soulful |
| `charming` | attitude | social | Charming and coy, a twinkle in the voice. | flirtatious, flirty, coy | style:act_coy |
| `whisper` | delivery | vocal | Whispering, breathy and unvoiced, very close to the ear. | whispering, whispered | style:whisper |
| `murmur` | delivery | vocal | Murmuring under the breath, low and private. | muttering, under_breath, sotto_voce, aside | style:murmur |
| `soft` | delivery | vocal | Soft and hushed, quiet but fully voiced. | quiet, hushed, softly | style:gentle |
| `breathy` | delivery | vocal | Breathy and airy, a lot of air in the voice. | airy | style:ethereal |
| `mumbling` | delivery | vocal | Mumbling, words half-formed and indistinct. | mumble, muffled | style:murmur |
| `projected` | delivery | vocal | Projected and booming, filling a big room. | loud, calling_out, booming, announcing | style:loudly |
| `shout` | delivery | vocal | Shouting at full voice across a distance. | shouting, yell, yelling, holler | style:shout |
| `scream` | delivery | vocal | Screaming, raw and at the very top of the voice. | screaming, shriek, shrieking | style:shout |
| `roar` | delivery | vocal | Roaring, deep, loud and thunderous. | bellow, bellowing | style:roar |
| `laughing` | delivery | vocal | Talking while laughing, words bubbling through laughter. | while_laughing, laughing_speech | emotion:happy |
| `sing_song` | delivery | vocal | Sing-song and lilting, melodic teasing rhythm. | singsong, lilting, chanting | style:vivid |

| nonverbal | Fish tag | Chatterbox Turbo | Step-Audio-EditX |
|---|---|---|---|
| `<laugh>` | [laughing] | [laugh] | [laugh] |
| `<chuckle>` | [chuckling] | [chuckle] | [chuckle] |
| `<giggle>` | [giggling] | - | [giggle] |
| `<sigh>` | [sigh] | [sigh] | [sigh] |
| `<gasp>` | [gasp] | [gasp] | - |
| `<sob>` | [sobbing] | - | - |
| `<sniff>` | [sniffs] | [sniff] | - |
| `<yawn>` | [yawning] | - | - |
| `<cough>` | [coughing] | [cough] | [cough] |
| `<clear_throat>` | [clears throat] | [clear throat] | [clears throat] |
| `<groan>` | [groaning] | [groan] | - |
| `<scream>` | [screaming] | - | - |
| `<whimper>` | [whimpering] | - | - |
| `<breath>` | [breath] | - | [breath] |
| `<inhale>` | [inhale] | - | [inhale] |
| `<exhale>` | [exhale] | - | [exhale] |
| `<pant>` | [panting] | - | - |
| `<hum>` | [humming] | - | - |
| `<snort>` | [snort] | - | [snort] |
| `<tsk>` | [tsk] | - | - |
| `<shush>` | [shushing] | [shush] | - |
| `<gulp>` | [gulp] | - | - |
| `<hmm>` | [hmm] | - | [uhm] |

| backend setup | native+bank | weighted | nonverbals |
|---|---|---|---|
| indextts | 45.8% | 72.9% | 0.0% |
| indextts + full emotion bank | 100.0% | 100.0% | 100.0% |
| fish (model=s2-pro) | 87.5% | 93.8% | 100.0% |
| fish (model=drama-3) | 100.0% | 100.0% | 100.0% |
| qwen3 (mode=custom_voice) | 87.5% | 93.8% | 0.0% |
| openai_speech (emotion_mode=inline_tags) | 87.5% | 93.8% | 100.0% |
| chatterbox (model=original) | 0.0% | 50.0% | 0.0% |
| chatterbox (model=turbo) | 0.0% | 50.0% | 39.1% |
| step_edit (base=indextts) | 59.4% | 79.7% | 47.8% |
| kokoro | 0.0% | 0.0% | 0.0% |
| espeak | 0.0% | 0.0% | 0.0% |
