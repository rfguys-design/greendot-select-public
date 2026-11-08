"""Skills table and response schema. OWNER: Joanna  (Week 3)

ONE model, MANY prompts. Read-aloud, solve, hint, translate, explain, copy and
plot are entries in this table -- not separate models, modules, or programs.
Adding a capability is adding a dict entry.

The model returns structured JSON against RESPONSE_SCHEMA, so the code never
parses prose.
"""

SKILLS = {
    "read_aloud": "Transcribe the text in the region exactly, word for word, as "
                  "spoken_text. Do not summarize, correct, or paraphrase.",
    "solve_math": "Work the problem in the region. Explain the steps briefly in "
                  "plain spoken sentences, then give the final answer.",
    "hint":       "The region contains a problem. Give ONE hint toward the next "
                  "step. Do not solve it and do not reveal the answer.",
    "translate":  "Translate ALL of source_text into the language the user asked "
                  "for (English if they did not say), sentence by sentence, in "
                  "order. Never summarize, shorten, or stop early. spoken_text is "
                  "ONLY the translation, written in that language's own script.",
    "explain":    "Explain the content of the region simply, under 120 words.",
    "copy_text":  "Transcribe the text in the region exactly for copying. Keep "
                  "the original line and paragraph breaks.",
    "plot_table": "The region contains a table. Extract it into the chart fields "
                  "and pick the chart type that suits the data. spoken_text is one "
                  "sentence describing what the chart shows.",
    "edit_image": "The user wants a NEW PICTURE made from the drawing or image in "
                  "the region: polish a sketch, change its style, or add or change "
                  "something in it. Do not describe the picture. spoken_text is one "
                  "short, friendly sentence saying what you are making, e.g. "
                  "'Sure, making your cat more hairy.' The picture itself is drawn "
                  "by a separate image model from the user's own words.",
    "unclear":    "Say briefly that the region could not be understood.",
}

# OpenAI text-to-speech voices. The model picks one when the user asks for a
# kind of speaker; the accent itself goes in voice_instructions.
VOICES = {
    "nova": "female, bright (default)",
    "coral": "female, warm",
    "shimmer": "female, soft",
    "sage": "female, calm",
    "ash": "male, clear",
    "onyx": "male, deep",
    "echo": "male, neutral",
    "fable": "male, expressive storyteller",
    "ballad": "male, gentle",
    "verse": "male, lively",
    "alloy": "neutral",
}

CHART_TYPES = ["bar", "line", "scatter", "pie"]

SYSTEM_PROMPT = (
    "You help a student working with a physical page. Two green dots mark "
    "opposite corners of a region; the image you receive is that region, "
    "already cropped.\n\n"
    "Decide which task the user wants from what they said and what is in the "
    "image, then produce the fields for that task.\n\n"
    "source_text: for read_aloud, translate and copy_text, FIRST transcribe "
    "every line of text in the region exactly, top to bottom, in its original "
    "language -- all of it, even when it is long. The other fields are built "
    "from it, so nothing may be missing. For other tasks leave it empty.\n\n"
    "Tasks:\n"
    + "\n".join(f"- {k}: {v}" for k, v in SKILLS.items())
    + "\n\nspoken_text will be read aloud by a speech synthesizer. Write plain "
    "sentences. No bullet points, asterisks, markdown, or LaTeX. Spell out "
    "symbols as words where needed. Be warm and concise.\n\n"
    "display_text is shown on the student's screen. For hint, solve_math and "
    "explain it may use simple math notation like x^2 and line breaks. For "
    "translate, give the translation. For read_aloud and copy_text, the exact "
    "text. Otherwise repeat spoken_text.\n\n"
    "Voice: choose voice from this list, matching any speaker the user asked "
    "for (man, woman, deep, ...); otherwise use nova.\n"
    + "\n".join(f"- {k}: {v}" for k, v in VOICES.items())
    + "\nvoice_instructions tells the speech synthesizer HOW to speak: accent, "
    "language, pace, tone. Put any accent or style the user asked for here, "
    "e.g. 'Speak with a British accent.' When spoken_text is not English, say "
    "which language, e.g. 'Speak in Mandarin Chinese.' Otherwise leave it empty.\n\n"
    "Chart fields are only for plot_table; for every other task leave the "
    "strings empty and the lists empty. x_values are the category or x-axis "
    "labels in table order; each series is one numeric column, with one value "
    "per x_value (use 0 for a blank cell)."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tutor_response",
        "strict": True,
        "schema": {
            "type": "object",
            # Order matters: the model writes fields in this order, so
            # source_text (the full transcription) exists before the
            # translation is written -- that is what stops it skipping text.
            "properties": {
                "task": {"type": "string", "enum": list(SKILLS.keys())},
                "source_text": {"type": "string"},
                "spoken_text": {"type": "string"},
                "display_text": {"type": "string"},
                "voice": {"type": "string", "enum": list(VOICES.keys())},
                "voice_instructions": {"type": "string"},
                "chart": {
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string", "enum": CHART_TYPES},
                        "title": {"type": "string"},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"},
                        "x_values": {"type": "array", "items": {"type": "string"}},
                        "series": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "values": {"type": "array", "items": {"type": "number"}},
                                },
                                "required": ["name", "values"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["chart_type", "title", "x_label", "y_label", "x_values", "series"],
                    "additionalProperties": False,
                },
                "confidence": {"type": "number"},
            },
            "required": ["task", "source_text", "spoken_text", "display_text", "voice",
                         "voice_instructions", "chart", "confidence"],
            "additionalProperties": False,
        },
    },
}
