"""Canonical emotion taxonomy for VoicingSoundscaping.

One backend-agnostic vocabulary for everything a performer can do with a voice:

* ``emotion``  - an affective state (joy, grief, fear ...)
* ``attitude`` - a social colour or stance (dry humor, cheeky, sarcastic, comforting ...)
* ``state``    - a physical condition that changes the voice (crying, sleepy, out of breath ...)
* ``delivery`` - a vocal mode, combinable with any of the above (whisper, shout, scream ...)

Every entry carries what each synthesis backend needs to realise it:

* ``vec``      - IndexTTS-2/2.5 emotion vector components
                 (order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)
* ``describe`` - a natural-language acting note (instruct backends: Qwen3-TTS, OpenAI-style servers)
* ``tag``      - a short free-form tag (Fish S2-Pro / Drama 3, Higgs-style inline tags)
* ``step``     - nearest Step-Audio-EditX edit, e.g. ``("emotion", "happy")`` or ``("style", "whisper")``
* ``rate``/``pitch``/``level`` - prosody hints at moderate intensity (0.6), used by DSP fallbacks,
                 speed controls and the mixer (whispers stay quiet, shouts stay big)
* ``line``     - a reference sentence used to bootstrap emotion banks (reference clips per emotion)
* ``hard``     - extreme vocalisations that instruct-only models tend to under-deliver

Intensity is always 0..1. ``0.6`` is "clearly present, moderate".
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("emotion", "attitude", "state", "delivery")
FAMILIES = (
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "calm", "low_energy", "social", "vocal",
)
INDEX_DIMS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")

STEP_EMOTIONS = (
    "happy", "angry", "sad", "fear", "surprised", "confusion", "empathy", "embarrass",
    "excited", "depressed", "admiration", "coldness", "disgusted", "humour",
)
STEP_STYLES = (
    "serious", "arrogant", "child", "older", "girl", "pure", "sister", "sweet", "exaggerated",
    "ethereal", "whisper", "generous", "recite", "act_coy", "warm", "shy", "comfort",
    "authority", "chat", "radio", "soulful", "gentle", "story", "vivid", "program", "news",
    "advertising", "roar", "murmur", "shout", "deeply", "loudly",
)
E2V_CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised")


@dataclass(frozen=True)
class Emotion:
    id: str
    kind: str
    family: str
    valence: float
    arousal: float
    dominance: float
    describe: str
    vec: tuple[float, ...]
    tag: str
    step: tuple[str, str] | None
    nonverbals: tuple[str, ...]
    rate: float
    pitch: float
    level: float
    aliases: tuple[str, ...]
    line: str
    e2v: str
    hard: bool = False

    @property
    def is_delivery(self) -> bool:
        return self.kind == "delivery"


@dataclass(frozen=True)
class Nonverbal:
    id: str
    describe: str
    aliases: tuple[str, ...] = ()
    fish: str = ""            # free-form tag for Fish S2-Pro / Drama 3
    chatterbox: str = ""      # native Chatterbox-Turbo/Nano tag (empty = unsupported)
    step: str = ""            # native Step-Audio-EditX paralinguistic tag (empty = unsupported)
    duration: float = 0.8     # typical length in seconds (for timing placeholders)
    tags: dict = field(default_factory=dict)


def _vec(spec: str) -> tuple[float, ...]:
    parts = dict(p.split(":") for p in spec.split()) if spec else {}
    unknown = set(parts) - set(INDEX_DIMS)
    if unknown:
        raise ValueError(f"unknown IndexTTS dims {unknown}")
    return tuple(float(parts.get(d, 0.0)) for d in INDEX_DIMS)


def _step(spec: str | None) -> tuple[str, str] | None:
    if not spec:
        return None
    kind, value = spec.split(":")
    allowed = STEP_EMOTIONS if kind == "emotion" else STEP_STYLES
    if kind not in ("emotion", "style") or value not in allowed:
        raise ValueError(f"bad Step-Audio-EditX mapping {spec}")
    return (kind, value)


_E2V_BY_DIM = {
    "happy": "happy", "angry": "angry", "sad": "sad", "afraid": "fearful",
    "disgusted": "disgusted", "melancholic": "sad", "surprised": "surprised", "calm": "neutral",
}


def E(id, kind, family, vad, describe, vec, *, step=None, nv="", rate=1.0, pitch=0.0, level=0.0,
      aliases="", line, tag=None, e2v=None, hard=False) -> Emotion:
    v = _vec(vec)
    if e2v is None:
        e2v = _E2V_BY_DIM[INDEX_DIMS[max(range(8), key=lambda i: v[i])]] if any(v) else "neutral"
    return Emotion(
        id=id, kind=kind, family=family, valence=vad[0], arousal=vad[1], dominance=vad[2],
        describe=describe, vec=v, tag=tag or id.replace("_", " "), step=_step(step),
        nonverbals=tuple(nv.split()), rate=rate, pitch=pitch, level=level,
        aliases=tuple(a for a in aliases.split() if a), line=line, e2v=e2v, hard=hard,
    )


EMOTIONS: tuple[Emotion, ...] = (
    # ------------------------------------------------------------------ joy
    E("joy", "emotion", "joy", (0.85, 0.6, 0.5),
      "Joyful and bright, smiling through every word.", "happy:0.8 surprised:0.1",
      step="emotion:happy", nv="laugh", rate=1.05, pitch=1.0, level=1, aliases="joyful happy glad happiness",
      line="We did it! We actually did it, and it's even more wonderful than I imagined!"),
    E("delight", "emotion", "joy", (0.8, 0.5, 0.4),
      "Delighted and charmed, a light lift of pleasant surprise.", "happy:0.7 surprised:0.3",
      step="emotion:happy", nv="giggle", rate=1.03, pitch=1.0, aliases="delighted charmed",
      line="Oh, how lovely! You remembered my favourite flowers."),
    E("excited", "emotion", "joy", (0.75, 0.85, 0.5),
      "Excited and eager, words tumbling out quickly with bright energy.", "happy:0.7 surprised:0.3",
      step="emotion:excited", rate=1.15, pitch=2.0, level=2, aliases="eager thrilled enthusiastic excitement",
      line="Come on, come on, hurry! It's starting, I can see the lights already!"),
    E("ecstatic", "emotion", "joy", (0.95, 0.95, 0.6),
      "Ecstatic and overjoyed, bursting with uncontainable happiness.", "happy:1.0 surprised:0.3",
      step="emotion:excited", nv="laugh", rate=1.15, pitch=3.0, level=3, aliases="elated overjoyed euphoric",
      line="Yes! Yes! I got in! I can't believe it, I actually got in!", hard=True),
    E("amused", "emotion", "joy", (0.6, 0.35, 0.4),
      "Amused and tickled, a chuckle sitting just under the words.", "happy:0.6 calm:0.2",
      step="emotion:humour", nv="chuckle", rate=1.0, pitch=0.5, aliases="tickled entertained amusement",
      line="Oh, that's rich. You really thought the goat would listen to you?"),
    E("playful", "attitude", "joy", (0.7, 0.55, 0.4),
      "Playful and light, bouncing rhythm, having fun.", "happy:0.6 surprised:0.2",
      step="emotion:humour", nv="giggle", rate=1.08, pitch=1.5, aliases="fun frolicsome",
      line="Catch me if you can! Come on, slowpoke!"),
    E("cheeky", "attitude", "joy", (0.55, 0.5, 0.6),
      "Cheeky and impish, a grin you can hear, teasing without malice.", "happy:0.5 surprised:0.1 calm:0.1",
      step="emotion:humour", nv="giggle chuckle", rate=1.05, pitch=1.0, aliases="impish mischievous sassy cheekily",
      line="Who, me? I would never eat the last cookie. Crumbs? What crumbs?"),
    E("teasing", "attitude", "joy", (0.5, 0.45, 0.6),
      "Teasing and affectionate, drawing out the words with a knowing smile.", "happy:0.5 calm:0.1",
      step="emotion:humour", nv="giggle", rate=1.0, pitch=1.0, aliases="ribbing joshing",
      line="Ooh, somebody's got a crush on the baker's boy."),
    E("giddy", "emotion", "joy", (0.8, 0.8, 0.3),
      "Giddy and bubbly, barely holding back giggles.", "happy:0.8 surprised:0.2",
      step="emotion:excited", nv="giggle", rate=1.12, pitch=2.0, level=1, aliases="giggly bubbly",
      line="I can't stop smiling, it's silly, I know, but I just can't!"),
    E("peppy", "attitude", "joy", (0.7, 0.75, 0.5),
      "Peppy and perky, upbeat sparkle, energetic and chipper.", "happy:0.7 surprised:0.1",
      step="style:vivid", rate=1.12, pitch=1.5, level=1, aliases="perky chipper upbeat peaky bouncy",
      line="Good morning, sunshine! Rise and shine, the pancakes won't flip themselves!"),
    E("proud", "emotion", "joy", (0.7, 0.4, 0.8),
      "Proud and warm, chest lifted, savouring the accomplishment.", "happy:0.6 calm:0.2",
      step="emotion:happy", rate=0.97, pitch=0.5, level=1, aliases="pride accomplished",
      line="I built it with my own two hands, every single plank."),
    E("triumphant", "emotion", "joy", (0.85, 0.85, 0.9),
      "Triumphant and victorious, ringing out with power.", "happy:0.8 surprised:0.1 angry:0.1",
      step="emotion:excited", rate=1.05, pitch=1.5, level=3, aliases="victorious exultant",
      line="At last! The dragon is defeated and the village is free!"),
    E("relieved", "emotion", "joy", (0.6, -0.2, 0.2),
      "Relieved, tension melting away, a long breath out.", "happy:0.4 calm:0.5",
      step="emotion:happy", nv="sigh exhale", rate=0.95, pitch=-0.5, aliases="relief",
      line="Oh, thank goodness. You're safe. You're really safe."),
    E("content", "emotion", "joy", (0.7, -0.4, 0.4),
      "Contented and cosy, unhurried, quietly happy.", "happy:0.4 calm:0.6",
      step="style:warm", rate=0.95, pitch=-0.5, aliases="contentment satisfied at_peace cozy cosy",
      line="A warm fire, a full belly, and good company. What more could anyone want?"),
    E("grateful", "emotion", "joy", (0.75, 0.1, 0.2),
      "Grateful and heartfelt, sincere thanks with a touch of emotion.", "happy:0.5 calm:0.3 melancholic:0.1",
      step="style:soulful", rate=0.95, aliases="thankful gratitude",
      line="Thank you. Truly. I don't know what I would have done without you."),
    E("tender", "emotion", "joy", (0.8, -0.2, 0.3),
      "Tender and loving, soft and close, full of affection.", "happy:0.4 calm:0.5 melancholic:0.1",
      step="style:gentle", rate=0.92, pitch=-0.5, level=-2, aliases="loving affectionate love fond",
      line="Sleep now, little one. I'll be right here when you wake."),
    E("hopeful", "emotion", "joy", (0.6, 0.2, 0.3),
      "Hopeful and lifting, a gentle brightness reaching forward.", "happy:0.5 calm:0.2 surprised:0.1",
      step="emotion:happy", rate=1.0, pitch=0.5, aliases="optimistic hope",
      line="Maybe, just maybe, the spring will come early this year."),
    E("warm", "attitude", "joy", (0.7, 0.1, 0.4),
      "Warm and friendly, welcoming, an open smile in the voice.", "happy:0.4 calm:0.5",
      step="style:warm", rate=0.98, aliases="friendly kind welcoming",
      line="Come in, come in, out of the cold. There's tea on the stove."),
    E("admiring", "emotion", "joy", (0.7, 0.3, 0.1),
      "Admiring and impressed, warm respect and a little wonder.", "happy:0.5 surprised:0.3",
      step="emotion:admiration", rate=0.97, aliases="admiration impressed respectful",
      line="You painted this? It's beautiful. It truly takes my breath away."),
    E("awe", "emotion", "surprise", (0.6, 0.5, -0.3),
      "Awestruck and full of wonder, hushed and slow, eyes wide.", "surprised:0.6 happy:0.3 calm:0.1",
      step="emotion:admiration", rate=0.9, level=-1, aliases="awestruck wonder amazed marvelling wonderstruck",
      line="Look at the sky... thousands of stars, more than I ever knew existed."),
    # ------------------------------------------------------------- sadness
    E("sad", "emotion", "sadness", (-0.7, -0.3, -0.4),
      "Sad and heavy, voice lowered, the words weighed down.", "sad:0.8",
      step="emotion:sad", nv="sigh", rate=0.92, pitch=-1.0, level=-1, aliases="unhappy down sorrowful sadness",
      line="I waited by the window all evening, but nobody came."),
    E("melancholic", "emotion", "sadness", (-0.5, -0.5, -0.3),
      "Melancholic and grey, a slow quiet ache.", "melancholic:0.8",
      step="emotion:sad", rate=0.9, pitch=-1.0, level=-1, aliases="melancholy gloomy blue",
      line="The old house is quiet now. Even the clocks seem to have stopped caring."),
    E("wistful", "emotion", "sadness", (-0.2, -0.4, -0.1),
      "Wistful and nostalgic, a bittersweet smile at a faraway memory.", "melancholic:0.5 happy:0.2 calm:0.3",
      step="style:soulful", rate=0.92, pitch=-0.5, aliases="nostalgic longing bittersweet yearning",
      line="We used to swim in that lake every summer. It feels like another life."),
    E("heartbroken", "emotion", "sadness", (-0.9, 0.2, -0.6),
      "Heartbroken and devastated, voice cracking with grief.", "sad:0.9 melancholic:0.3",
      step="emotion:sad", nv="sob sniff", rate=0.88, pitch=-0.5, level=-1, aliases="grief grieving devastated anguished",
      line="She's gone. She's really gone, and I never got to say goodbye.", hard=True),
    E("crying", "state", "sadness", (-0.85, 0.4, -0.6),
      "Crying, speaking through tears, voice breaking and wet, catching breath between words.", "sad:1.0",
      step="emotion:sad", nv="sob sniff", rate=0.85, pitch=0.5, level=-1, aliases="cry tearful weeping through_tears teary",
      line="I'm sorry... I just... I miss him so much.", hard=True),
    E("sobbing", "delivery", "sadness", (-0.9, 0.7, -0.7),
      "Sobbing, words broken by heaving sobs and ragged breaths.", "sad:1.0 afraid:0.1",
      step="emotion:sad", nv="sob", rate=0.8, pitch=1.0, level=0, aliases="sobs bawling wailing",
      line="Please... please don't go... please...", hard=True),
    E("despair", "emotion", "sadness", (-0.95, 0.3, -0.9),
      "Despairing and hopeless, the bottom has fallen out.", "sad:0.8 melancholic:0.5 afraid:0.2",
      step="emotion:depressed", rate=0.88, pitch=-1.0, aliases="hopeless despairing desperate",
      line="There's no point anymore. Whatever we do, it all falls apart."),
    E("lonely", "emotion", "sadness", (-0.6, -0.5, -0.5),
      "Lonely and small, quiet and a little hollow.", "melancholic:0.7 sad:0.3",
      step="emotion:sad", rate=0.9, pitch=-0.5, level=-2, aliases="alone isolated",
      line="Another birthday, another empty chair across the table."),
    E("disappointed", "emotion", "sadness", (-0.5, -0.1, -0.2),
      "Disappointed and deflated, the hope draining out of the sentence.", "sad:0.5 melancholic:0.3",
      step="emotion:sad", nv="sigh", rate=0.94, pitch=-1.0, aliases="let_down deflated disappointment",
      line="Oh. I see. I thought you'd be there this time."),
    E("regretful", "emotion", "sadness", (-0.6, -0.2, -0.4),
      "Regretful and remorseful, heavy with what cannot be undone.", "sad:0.5 melancholic:0.4",
      step="emotion:sad", rate=0.9, pitch=-0.5, aliases="remorseful rueful regret",
      line="I should have listened to you. I'm so sorry I didn't."),
    E("hurt", "emotion", "sadness", (-0.7, 0.3, -0.5),
      "Hurt and wounded, a sting of betrayal in the voice.", "sad:0.6 angry:0.2",
      step="emotion:sad", rate=0.95, aliases="wounded stung",
      line="You said that in front of everyone. Do you know how that felt?"),
    E("depressed", "emotion", "sadness", (-0.8, -0.7, -0.6),
      "Dejected and flat, low energy, colour drained from the voice.", "melancholic:0.6 sad:0.5",
      step="emotion:depressed", rate=0.85, pitch=-2.0, level=-2, aliases="dejected discouraged downcast",
      line="I don't know. It doesn't matter. Nothing really does."),
    E("sympathetic", "emotion", "sadness", (0.3, -0.2, 0.0),
      "Sympathetic and compassionate, gentle understanding.", "sad:0.3 calm:0.5",
      step="emotion:empathy", rate=0.93, level=-1, aliases="empathetic compassionate understanding empathy",
      line="That sounds so hard. I'm really sorry you went through that alone."),
    # --------------------------------------------------------------- anger
    E("angry", "emotion", "anger", (-0.6, 0.8, 0.7),
      "Angry and sharp, hard consonants, heat in every word.", "angry:0.8",
      step="emotion:angry", rate=1.05, pitch=0.5, level=2, aliases="mad anger cross",
      line="How dare you! That was not yours to take!"),
    E("furious", "emotion", "anger", (-0.8, 0.95, 0.9),
      "Furious and enraged, explosive, barely contained.", "angry:1.0",
      step="emotion:angry", rate=1.08, pitch=1.0, level=3, aliases="rage enraged livid seething wrathful",
      line="Get out! Get out of my house and never come back!", hard=True),
    E("irritated", "emotion", "anger", (-0.4, 0.4, 0.4),
      "Irritated and annoyed, clipped and impatient.", "angry:0.4 disgusted:0.1",
      step="emotion:angry", nv="tsk sigh", rate=1.03, aliases="annoyed irked bothered",
      line="Must you hum that same song every single morning?"),
    E("frustrated", "emotion", "anger", (-0.5, 0.6, 0.1),
      "Frustrated and exasperated, at the very end of one's patience.", "angry:0.5 sad:0.2",
      step="emotion:angry", nv="groan sigh", rate=1.05, pitch=0.5, level=1, aliases="exasperated fed_up frustration",
      line="I've tried it ten times! Why won't this silly knot just come loose?"),
    E("peeved", "emotion", "anger", (-0.35, 0.3, 0.4),
      "Peeved and huffy, mildly offended, pouting a little.", "angry:0.35 disgusted:0.15",
      step="emotion:angry", nv="tsk", rate=1.0, aliases="miffed put_out huffy sulky",
      line="Well, fine. Nobody asked me, apparently. As usual."),
    E("indignant", "emotion", "anger", (-0.5, 0.6, 0.6),
      "Indignant and affronted, righteous outrage.", "angry:0.6 surprised:0.2",
      step="emotion:angry", rate=1.05, pitch=1.0, level=1, aliases="outraged affronted",
      line="Me? A thief? I have never stolen anything in my life!"),
    E("bitter", "emotion", "anger", (-0.6, 0.2, 0.3),
      "Bitter and resentful, sour edges on the words.", "angry:0.3 sad:0.3 disgusted:0.3",
      step="emotion:coldness", rate=0.95, pitch=-0.5, aliases="resentful sour",
      line="Of course they chose him. They always choose him."),
    E("contemptuous", "attitude", "disgust", (-0.6, 0.1, 0.8),
      "Contemptuous and scornful, looking down the nose.", "disgusted:0.6 angry:0.3",
      step="emotion:disgusted", rate=0.95, aliases="scornful disdainful sneering contempt",
      line="You call that a sword? My grandmother carves better with a butter knife."),
    E("defiant", "attitude", "anger", (-0.2, 0.7, 0.9),
      "Defiant and unbowed, firm and unyielding.", "angry:0.5 calm:0.1",
      step="style:authority", rate=1.0, level=1, aliases="rebellious stubborn unyielding",
      line="No. I won't bow, and you can't make me."),
    E("menacing", "attitude", "anger", (-0.6, 0.3, 0.9),
      "Quietly menacing, low and controlled, a threat beneath every word.", "angry:0.5 calm:0.3 disgusted:0.2",
      step="style:deeply", rate=0.9, pitch=-1.5, level=-1, aliases="threatening sinister villainous ominous",
      line="Run along now, little mouse. Before I change my mind."),
    E("grumpy", "emotion", "anger", (-0.4, 0.2, 0.3),
      "Grumpy and grouchy, gravelly grumbling.", "angry:0.3 disgusted:0.2 melancholic:0.1",
      step="emotion:coldness", nv="groan", rate=0.95, pitch=-1.0, aliases="cranky grouchy curmudgeonly",
      line="Bah. Children, fireworks, and singing. Wake me when it's over."),
    E("jealous", "emotion", "anger", (-0.5, 0.4, 0.0),
      "Jealous and envious, a sulky edge.", "angry:0.4 sad:0.3 disgusted:0.1",
      step="emotion:angry", rate=1.0, aliases="envious envy jealousy",
      line="Why does she always get the shiny new things? It isn't fair."),
    # ---------------------------------------------------------------- fear
    E("afraid", "emotion", "fear", (-0.7, 0.6, -0.7),
      "Afraid and frightened, voice tight and unsteady.", "afraid:0.8",
      step="emotion:fear", rate=1.05, pitch=1.0, aliases="scared fearful frightened fear",
      line="Did you hear that? Something is moving in the dark."),
    E("terrified", "emotion", "fear", (-0.9, 0.95, -0.9),
      "Terrified, raw panic, high and breathless.", "afraid:1.0 surprised:0.2",
      step="emotion:fear", nv="gasp whimper", rate=1.15, pitch=2.0, level=1, aliases="horrified petrified terror",
      line="It's coming! Run! Run!", hard=True),
    E("anxious", "emotion", "fear", (-0.5, 0.5, -0.5),
      "Anxious and nervous, jittery rhythm, slightly rushed.", "afraid:0.5 sad:0.1",
      step="emotion:fear", rate=1.08, pitch=0.5, aliases="nervous uneasy jittery on_edge anxiety",
      line="What if they don't like me? What if I say the wrong thing?"),
    E("worried", "emotion", "fear", (-0.5, 0.3, -0.4),
      "Worried and concerned, a furrowed-brow tone.", "afraid:0.4 sad:0.3",
      step="emotion:fear", rate=1.0, aliases="concerned troubled worry",
      line="It's nearly midnight. He should have been home hours ago."),
    E("panicked", "emotion", "fear", (-0.8, 0.95, -0.8),
      "Panicked and frantic, tumbling words, gasping for air.", "afraid:0.9 surprised:0.3",
      step="emotion:fear", nv="pant gasp", rate=1.25, pitch=2.0, level=1, aliases="frantic alarmed panic",
      line="Where is it, where is it, I had it right here, it was right here!", hard=True),
    E("suspicious", "attitude", "fear", (-0.3, 0.2, 0.4),
      "Suspicious and wary, narrowed eyes, measured and probing.", "afraid:0.2 disgusted:0.2 calm:0.3",
      step="emotion:coldness", rate=0.95, pitch=-0.5, aliases="wary distrustful skeptical sceptical",
      line="And why exactly would a stranger give away a magic bean for free?"),
    E("trembling", "delivery", "fear", (-0.6, 0.5, -0.7),
      "Voice trembling and unsteady, barely controlled.", "afraid:0.6 sad:0.3",
      step="emotion:fear", rate=0.95, pitch=1.0, level=-1, aliases="shaky quavering voice_shaking",
      line="I... I'm not scared. I'm just... cold. That's all.", hard=True),
    # ------------------------------------------------------------ surprise
    E("surprised", "emotion", "surprise", (0.2, 0.7, 0.0),
      "Surprised, caught off guard, pitch jumping up.", "surprised:0.8",
      step="emotion:surprised", nv="gasp", rate=1.05, pitch=2.0, level=1, aliases="surprise startled",
      line="You're here? But you said you couldn't come!"),
    E("shocked", "emotion", "surprise", (-0.4, 0.85, -0.3),
      "Shocked and stunned, disbelief knocking the wind out.", "surprised:0.8 afraid:0.3",
      step="emotion:surprised", nv="gasp", rate=1.0, pitch=1.5, level=1, aliases="stunned aghast shock",
      line="What? The bridge is gone? The whole bridge?"),
    E("confused", "emotion", "surprise", (-0.2, 0.2, -0.3),
      "Confused and puzzled, searching, rising question marks.", "surprised:0.4 calm:0.2",
      step="emotion:confusion", nv="hmm", rate=0.95, pitch=0.5, aliases="puzzled bewildered perplexed confusion",
      line="Wait, so the map is upside down? Then where have we been walking?"),
    E("curious", "emotion", "surprise", (0.4, 0.4, 0.2),
      "Curious and intrigued, leaning in, a playful question in the voice.", "surprised:0.4 happy:0.3",
      step="emotion:happy", rate=1.0, pitch=0.5, aliases="intrigued inquisitive curiosity",
      line="What's behind this little door, I wonder? Shall we take a peek?"),
    # ------------------------------------------------------------- disgust
    E("disgusted", "emotion", "disgust", (-0.7, 0.4, 0.4),
      "Disgusted and grossed out, recoiling.", "disgusted:0.8",
      step="emotion:disgusted", nv="groan", rate=1.0, aliases="grossed_out repulsed disgust revolted",
      line="Ugh, it's slimy! And it smells like old socks!"),
    # -------------------------------------------------------- calm / neutral
    E("neutral", "emotion", "calm", (0.0, 0.0, 0.0),
      "Neutral and natural, clear and even.", "calm:0.3",
      rate=1.0, aliases="plain normal default", e2v="neutral",
      line="The train leaves at seven, and the journey takes about three hours."),
    E("calm", "emotion", "calm", (0.3, -0.6, 0.3),
      "Calm and steady, grounded, unhurried.", "calm:0.9",
      step="style:gentle", rate=0.94, pitch=-0.5, aliases="composed steady",
      line="Breathe slowly. Everything is going to be all right."),
    E("serene", "emotion", "calm", (0.6, -0.7, 0.3),
      "Serene and peaceful, soft and spacious.", "calm:0.8 happy:0.2",
      step="style:ethereal", rate=0.9, pitch=-0.5, level=-1, aliases="peaceful tranquil",
      line="The lake is perfectly still this morning, like a mirror for the clouds."),
    E("storytelling", "attitude", "calm", (0.3, 0.1, 0.4),
      "Engaged storytelling narration, warm, clear and well-paced, drawing the listener in.", "calm:0.6 happy:0.2",
      step="style:story", rate=0.97, aliases="narration narrative storyteller narrator",
      line="Once upon a time, in a village at the edge of a great forest, there lived a girl named Pip."),
    E("solemn", "attitude", "calm", (-0.2, -0.3, 0.5),
      "Solemn and serious, measured, carrying weight.", "calm:0.6 melancholic:0.3",
      step="style:serious", rate=0.9, pitch=-1.0, aliases="serious grave formal",
      line="Today we remember those who gave everything, so that we might be here."),
    E("mysterious", "attitude", "calm", (0.0, 0.1, 0.6),
      "Mysterious and hushed, as if sharing an ancient secret.", "calm:0.5 surprised:0.2 afraid:0.1",
      step="style:murmur", rate=0.9, pitch=-0.5, level=-2, aliases="enigmatic cryptic conspiratorial secretive",
      line="Few know the old door beneath the willow. Fewer still know what it opens."),
    E("dreamy", "emotion", "calm", (0.5, -0.6, -0.2),
      "Dreamy and faraway, drifting, soft focus.", "calm:0.6 happy:0.3",
      step="style:ethereal", rate=0.9, level=-1, aliases="faraway daydreaming wistful_dreamy",
      line="Imagine floating above the clouds, drifting wherever the wind goes..."),
    # ---------------------------------------------------------- low energy
    E("bored", "emotion", "low_energy", (-0.3, -0.7, 0.1),
      "Bored and unimpressed, flat and monotone, dragging the words.", "calm:0.4 melancholic:0.3 disgusted:0.1",
      step="emotion:coldness", nv="sigh yawn", rate=0.88, pitch=-1.0, level=-1, aliases="uninterested unimpressed monotone boredom",
      line="Yes, yes, another rainbow. Very nice. Can we go home now?"),
    E("indifferent", "attitude", "low_energy", (-0.1, -0.5, 0.4),
      "Indifferent and detached, cool and uninvested.", "calm:0.6 disgusted:0.1",
      step="emotion:coldness", rate=0.95, aliases="apathetic detached cold aloof",
      line="Do what you like. It makes no difference to me."),
    E("tired", "state", "low_energy", (-0.2, -0.7, -0.3),
      "Tired and weary, heavy and slow, low on breath.", "calm:0.5 melancholic:0.3 sad:0.1",
      step="emotion:depressed", nv="yawn sigh", rate=0.88, pitch=-1.0, level=-1, aliases="weary fatigued worn_out",
      line="It's been such a long day. My feet feel like they're made of stone."),
    E("exhausted", "state", "low_energy", (-0.4, -0.85, -0.5),
      "Exhausted and drained, every word an effort.", "melancholic:0.4 sad:0.3 calm:0.3",
      step="emotion:depressed", nv="sigh pant", rate=0.8, pitch=-1.5, level=-2, aliases="drained spent",
      line="I can't... I can't take another step. Just let me sit for a minute."),
    E("sleepy", "state", "low_energy", (0.1, -0.9, -0.3),
      "Sleepy and drowsy, words slow and slurred at the edges, fighting a yawn.", "calm:0.7 melancholic:0.2",
      step="style:murmur", nv="yawn", rate=0.78, pitch=-1.0, level=-3, aliases="drowsy yawning half_asleep",
      line="Mmm... just five more minutes... the bed is so warm..."),
    E("unwell", "state", "low_energy", (-0.5, -0.5, -0.5),
      "Unwell and congested, croaky and low on energy.", "sad:0.3 melancholic:0.3 calm:0.2",
      nv="cough sniff", rate=0.9, pitch=-1.0, level=-2, aliases="sick ill congested poorly",
      line="I think I'll stay in bed today. My head feels like a drum."),
    E("out_of_breath", "state", "vocal", (0.0, 0.8, -0.2),
      "Out of breath, panting between phrases after running.", "afraid:0.2 surprised:0.2",
      nv="pant", rate=0.95, pitch=1.0, aliases="panting breathless winded",
      line="Wait... wait for me... I ran... the whole way...", hard=True),
    # -------------------------------------------------------------- social
    E("dry_humor", "attitude", "social", (0.2, -0.4, 0.6),
      "Bone-dry deadpan wit, understated and unhurried; the joke lives in what is not emphasised.", "calm:0.7 happy:0.2",
      step="emotion:humour", rate=0.95, pitch=-0.5, aliases="deadpan dry wry droll dry_humour", tag="deadpan, dry wit",
      e2v="neutral", line="Oh good. Rain. On the one day I left my umbrella at home. Splendid."),
    E("sarcastic", "attitude", "social", (-0.3, 0.2, 0.6),
      "Sarcastic and ironic, exaggerated sweetness with a sharp edge.", "disgusted:0.3 happy:0.2 calm:0.3",
      step="emotion:humour", rate=0.97, aliases="ironic sardonic mocking sarcasm", e2v="other",
      line="Oh, brilliant idea. Let's poke the sleeping bear. What could possibly go wrong?"),
    E("smug", "attitude", "social", (0.4, 0.2, 0.9),
      "Smug and self-satisfied, savouring being right.", "happy:0.4 disgusted:0.2 calm:0.3",
      step="style:arrogant", rate=0.97, aliases="self_satisfied cocky gloating",
      line="I told you the shortcut would work. I'm always right, you know."),
    E("arrogant", "attitude", "social", (0.0, 0.3, 1.0),
      "Arrogant and haughty, grand and condescending.", "disgusted:0.3 calm:0.4 happy:0.1",
      step="style:arrogant", rate=0.95, pitch=-0.5, level=1, aliases="haughty condescending pompous",
      line="Kneel before the rightful king of the entire marsh!"),
    E("shy", "attitude", "social", (0.1, -0.2, -0.8),
      "Shy and bashful, small voice, hesitant.", "afraid:0.3 calm:0.3 happy:0.1",
      step="style:shy", rate=0.92, pitch=0.5, level=-3, aliases="bashful timid meek",
      line="Um... hello. I, um, I made you a card. You don't have to like it."),
    E("embarrassed", "emotion", "social", (-0.3, 0.4, -0.6),
      "Embarrassed and flustered, a nervous laugh at the edges.", "afraid:0.3 surprised:0.2 happy:0.1",
      step="emotion:embarrass", nv="giggle", rate=1.05, pitch=0.5, aliases="flustered sheepish mortified",
      line="Oh no, did I say that out loud? Please forget I said anything."),
    E("apologetic", "attitude", "social", (-0.2, -0.1, -0.6),
      "Apologetic and contrite, soft and sincere.", "sad:0.4 calm:0.3",
      step="emotion:empathy", rate=0.95, level=-1, aliases="sorry contrite",
      line="I'm so sorry about the vase. I'll make it up to you, I promise."),
    E("pleading", "attitude", "social", (-0.4, 0.5, -0.8),
      "Pleading and imploring, urgent and vulnerable.", "sad:0.5 afraid:0.3",
      step="emotion:sad", rate=1.0, pitch=1.0, aliases="begging imploring beseeching",
      line="Please, just one more chance. I know I can do it this time."),
    E("comforting", "attitude", "social", (0.6, -0.4, 0.4),
      "Comforting and reassuring, soothing and close.", "calm:0.6 happy:0.3",
      step="style:comfort", rate=0.9, pitch=-0.5, level=-2, aliases="reassuring soothing consoling",
      line="Shh, it's all right. It was just a bad dream. I'm here."),
    E("gentle", "attitude", "social", (0.5, -0.5, 0.2),
      "Gentle and soft-spoken, kind and careful.", "calm:0.7 happy:0.2",
      step="style:gentle", rate=0.93, level=-2, aliases="soft_spoken kindly",
      line="There we go, little bird. Back to your nest."),
    E("authoritative", "attitude", "social", (0.1, 0.4, 1.0),
      "Authoritative and commanding, firm and clear.", "calm:0.5 angry:0.2",
      step="style:authority", rate=0.97, pitch=-0.5, level=1, aliases="commanding stern firm",
      line="Everyone, stay together and keep to the path. No exceptions."),
    E("determined", "attitude", "social", (0.3, 0.6, 0.8),
      "Determined and resolute, steely focus.", "angry:0.3 calm:0.4 happy:0.1",
      step="style:authority", rate=1.0, level=1, aliases="resolute steely focused",
      line="We will reach the summit before nightfall. Whatever it takes."),
    E("sincere", "attitude", "social", (0.4, 0.0, 0.2),
      "Sincere and earnest, open-hearted.", "calm:0.6 happy:0.2",
      step="style:soulful", rate=0.95, aliases="earnest heartfelt honest",
      line="I mean it. You are the best friend I have ever had."),
    E("charming", "attitude", "social", (0.6, 0.4, 0.5),
      "Charming and coy, a twinkle in the voice.", "happy:0.5 calm:0.2",
      step="style:act_coy", rate=1.0, pitch=0.5, aliases="flirtatious flirty coy",
      line="Well, aren't you the gallant one, rescuing my hat from the river."),
    # ----------------------------------------------------------- deliveries
    E("whisper", "delivery", "vocal", (0.0, -0.4, -0.2),
      "Whispering, breathy and unvoiced, very close to the ear.", "calm:0.3 afraid:0.1",
      step="style:whisper", rate=0.95, level=-8, aliases="whispering whispered",
      line="Shh. If we're very quiet, we might see the fox cubs come out."),
    E("murmur", "delivery", "vocal", (0.0, -0.5, -0.2),
      "Murmuring under the breath, low and private.", "calm:0.5",
      step="style:murmur", rate=0.95, pitch=-0.5, level=-5, aliases="muttering under_breath sotto_voce aside",
      line="Not that anyone ever listens to me, of course."),
    E("soft", "delivery", "vocal", (0.2, -0.4, 0.0),
      "Soft and hushed, quiet but fully voiced.", "calm:0.6",
      step="style:gentle", rate=0.95, level=-4, aliases="quiet hushed softly",
      line="The baby's finally asleep. Let's tiptoe out."),
    E("breathy", "delivery", "vocal", (0.2, 0.0, -0.2),
      "Breathy and airy, a lot of air in the voice.", "calm:0.4 surprised:0.1",
      step="style:ethereal", level=-3, aliases="airy",
      line="Oh... it's beautiful up here. I can hardly breathe."),
    E("mumbling", "delivery", "vocal", (-0.2, -0.4, -0.4),
      "Mumbling, words half-formed and indistinct.", "calm:0.4 melancholic:0.2",
      step="style:murmur", rate=0.95, level=-4, aliases="mumble muffled",
      line="Mm, it's fine, I'll do it later, whatever."),
    E("projected", "delivery", "vocal", (0.2, 0.5, 0.7),
      "Projected and booming, filling a big room.", "calm:0.3 happy:0.2",
      step="style:loudly", rate=0.97, level=3, aliases="loud calling_out booming announcing",
      line="Ladies and gentlemen, welcome to the greatest show on the river!"),
    E("shout", "delivery", "vocal", (-0.1, 0.9, 0.8),
      "Shouting at full voice across a distance.", "angry:0.4 surprised:0.3",
      step="style:shout", rate=1.0, pitch=2.0, level=5, aliases="shouting yell yelling holler",
      line="Over here! We're over here! Can you hear me?"),
    E("scream", "delivery", "vocal", (-0.6, 1.0, -0.2),
      "Screaming, raw and at the very top of the voice.", "afraid:0.8 surprised:0.4",
      step="style:shout", nv="scream", rate=1.05, pitch=4.0, level=6, aliases="screaming shriek shrieking",
      line="Aaah! Help! Somebody help me!", hard=True),
    E("roar", "delivery", "vocal", (-0.5, 1.0, 1.0),
      "Roaring, deep, loud and thunderous.", "angry:0.9",
      step="style:roar", rate=0.95, pitch=-1.0, level=6, aliases="bellow bellowing",
      line="Who dares enter my cave?!", hard=True),
    E("laughing", "delivery", "vocal", (0.8, 0.7, 0.4),
      "Talking while laughing, words bubbling through laughter.", "happy:0.9",
      step="emotion:happy", nv="laugh", rate=1.05, pitch=1.5, level=1, aliases="while_laughing laughing_speech",
      line="Stop, stop, I can't breathe, you look ridiculous in that hat!", hard=True),
    E("sing_song", "delivery", "vocal", (0.6, 0.4, 0.4),
      "Sing-song and lilting, melodic teasing rhythm.", "happy:0.6",
      step="style:vivid", rate=1.0, pitch=1.5, aliases="singsong lilting chanting",
      line="I know a secret, and you don't, and I'm not telling!"),
)

NONVERBALS: tuple[Nonverbal, ...] = (
    Nonverbal("laugh", "a natural laugh", ("laughs", "laughter", "haha"), fish="[laughing]",
              chatterbox="[laugh]", step="[laugh]", duration=1.2),
    Nonverbal("chuckle", "a soft chuckle", ("chuckles",), fish="[chuckling]", chatterbox="[chuckle]",
              step="[chuckle]", duration=0.7),
    Nonverbal("giggle", "a giggle", ("giggles",), fish="[giggling]", step="[giggle]", duration=0.8),
    Nonverbal("sigh", "a sigh", ("sighs",), fish="[sigh]", chatterbox="[sigh]", step="[sigh]", duration=0.9),
    Nonverbal("gasp", "a sharp gasp", ("gasps",), fish="[gasp]", chatterbox="[gasp]", duration=0.5),
    Nonverbal("sob", "a sob", ("sobs",), fish="[sobbing]", duration=1.0),
    Nonverbal("sniff", "a sniffle", ("sniffs", "sniffle"), fish="[sniffs]", chatterbox="[sniff]", duration=0.5),
    Nonverbal("yawn", "a yawn", ("yawns",), fish="[yawning]", duration=1.4),
    Nonverbal("cough", "a cough", ("coughs",), fish="[coughing]", chatterbox="[cough]", step="[cough]", duration=0.6),
    Nonverbal("clear_throat", "clearing the throat", ("ahem", "clears_throat"), fish="[clears throat]",
              chatterbox="[clear throat]", step="[clears throat]", duration=0.7),
    Nonverbal("groan", "a groan", ("groans",), fish="[groaning]", chatterbox="[groan]", duration=0.9),
    Nonverbal("scream", "a scream", ("screams", "shriek"), fish="[screaming]", duration=1.2),
    Nonverbal("whimper", "a small whimper", ("whimpers",), fish="[whimpering]", duration=0.7),
    Nonverbal("breath", "an audible breath", ("breathe",), fish="[breath]", step="[breath]", duration=0.5),
    Nonverbal("inhale", "a sharp inhale", ("inhales",), fish="[inhale]", step="[inhale]", duration=0.5),
    Nonverbal("exhale", "a long exhale", ("exhales",), fish="[exhale]", step="[exhale]", duration=0.7),
    Nonverbal("pant", "panting", ("pants", "panting"), fish="[panting]", duration=1.0),
    Nonverbal("hum", "a short hum", ("hums",), fish="[humming]", duration=1.0),
    Nonverbal("snort", "a snort", ("snorts",), fish="[snort]", step="[snort]", duration=0.4),
    Nonverbal("tsk", "a disapproving tsk", ("tut",), fish="[tsk]", duration=0.3),
    Nonverbal("shush", "a shushing sound", ("shh",), fish="[shushing]", chatterbox="[shush]", duration=0.6),
    Nonverbal("gulp", "a nervous gulp", ("gulps",), fish="[gulp]", duration=0.4),
    Nonverbal("hmm", "a thoughtful hmm", ("uhm", "um", "hm"), fish="[hmm]", step="[uhm]", duration=0.6),
)

_BY_ID = {e.id: e for e in EMOTIONS}
_ALIASES: dict[str, str] = {}
for _e in EMOTIONS:
    for _a in (_e.id, *_e.aliases):
        _ALIASES.setdefault(_a.lower(), _e.id)
_NV_BY_ID = {n.id: n for n in NONVERBALS}
_NV_ALIASES: dict[str, str] = {}
for _n in NONVERBALS:
    for _a in (_n.id, *_n.aliases):
        _NV_ALIASES.setdefault(_a.lower(), _n.id)


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get(name: str) -> Emotion:
    """Look up an emotion by id or alias (case/space/hyphen-insensitive)."""
    key = _norm(name)
    if key in _ALIASES:
        return _BY_ID[_ALIASES[key]]
    raise KeyError(f"unknown emotion '{name}'")


def find(name: str) -> Emotion | None:
    try:
        return get(name)
    except KeyError:
        return None


def get_nonverbal(name: str) -> Nonverbal:
    key = _norm(name)
    if key in _NV_ALIASES:
        return _NV_BY_ID[_NV_ALIASES[key]]
    raise KeyError(f"unknown nonverbal '{name}'")


def find_nonverbal(name: str) -> Nonverbal | None:
    try:
        return get_nonverbal(name)
    except KeyError:
        return None


def suggest(name: str, n: int = 3) -> list[str]:
    import difflib

    return difflib.get_close_matches(_norm(name), list(_ALIASES) + list(_NV_ALIASES), n=n, cutoff=0.6)


def all_ids() -> list[str]:
    return [e.id for e in EMOTIONS]
