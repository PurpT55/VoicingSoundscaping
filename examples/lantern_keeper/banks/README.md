# Banks

* `emotions/<emotion_id>.wav` (+ `.txt` transcript): shared emotion references. IndexTTS uses them as
  `emo_audio_prompt`, so they may be performed by ANY speaker. Fill them with `vss bank build
  examples/lantern_keeper --backend fish` (or record family members acting each line from
  `vss emotions --list`). Numbered variants (`joy_1.wav`, `joy_2.wav`) are rotated.
* `nonverbals/<nonverbal_id>.wav`: laugh, sigh, sob, breath, yawn... clips spliced in for backends
  that cannot produce them (`<laugh>` in the script).

`vss bank list examples/lantern_keeper -v` shows what is missing.
