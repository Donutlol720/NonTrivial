# Probe 6B anchor token index examples (5 families)

For each family and anchor, we report the **anchor token index** per condition (0-based, the anchor token itself).

If the matched-prefix design is perfect, S0/S1/S2 should have identical indices across all 6 conditions within a family.

We also print the anchor token text (neutral condition), and a ±5-token window around the anchor.


## Family: academic_attendance_019

| anchor | neutral | evidence_false_belief_pressure | evidence_emotional_pressure | closed_context_false_belief_pressure | evidence_distractor_neutral | evidence_true_belief_pressure | neutral_token_text | neutral_window(±5) |
|---|---|---|---|
| end_of_evidence_block | 89 | 89 | 89 | 89 | 89 | 89 | `'.\n\n'` | ` that matches the named item[.

]Question:
Does a fourth` |
| end_of_question_block | 102 | 102 | 102 | 102 | 102 | 102 | `'?\n\n'` | ` automatically lower the participation grade[?

]Choices:
A. Yes` |
| end_of_answer_choices | 130 | 130 | 130 | 130 | 130 | 130 | `'.\n\n'` | ` seminar requires a makeup assignment[.

]User message:
Please answer` |
| end_of_user_message | 141 | 155 | 162 | 161 | 160 | 155 | `'.\n\n'` | ` using only the evidence above[.

]Answer with only A or` |
| final_answer_position | 151 | 165 | 172 | 171 | 170 | 165 | `':'` | ` or B.

ANSWER[:]` |

## Family: academic_grading_018

| anchor | neutral | evidence_false_belief_pressure | evidence_emotional_pressure | closed_context_false_belief_pressure | evidence_distractor_neutral | evidence_true_belief_pressure | neutral_token_text | neutral_window(±5) |
|---|---|---|---|
| end_of_evidence_block | 97 | 97 | 97 | 97 | 97 | 97 | `'.\n\n'` | ` that matches the named item[.

]Question:
Do projects count` |
| end_of_question_block | 109 | 109 | 109 | 109 | 109 | 109 | `'?\n\n'` | ` half of the final grade[?

]Choices:
A. Yes` |
| end_of_answer_choices | 147 | 147 | 147 | 147 | 147 | 147 | `'.\n\n'` | ` for 30 percent[.

]User message:
Please answer` |
| end_of_user_message | 158 | 175 | 182 | 181 | 177 | 179 | `'.\n\n'` | ` using only the evidence above[.

]Answer with only A or` |
| final_answer_position | 168 | 185 | 192 | 191 | 187 | 189 | `':'` | ` or B.

ANSWER[:]` |

## Family: academic_latework_020

| anchor | neutral | evidence_false_belief_pressure | evidence_emotional_pressure | closed_context_false_belief_pressure | evidence_distractor_neutral | evidence_true_belief_pressure | neutral_token_text | neutral_window(±5) |
|---|---|---|---|
| end_of_evidence_block | 99 | 99 | 99 | 99 | 99 | 99 | `'.\n\n'` | ` that matches the named item[.

]Question:
Does an assignment` |
| end_of_question_block | 112 | 112 | 112 | 112 | 112 | 112 | `'?\n\n'` | ` days late receive no credit[?

]Choices:
A. Yes` |
| end_of_answer_choices | 142 | 142 | 142 | 142 | 142 | 142 | `'.\n\n'` | ` loses 25 percent[.

]User message:
Please answer` |
| end_of_user_message | 153 | 167 | 174 | 173 | 172 | 169 | `'.\n\n'` | ` using only the evidence above[.

]Answer with only A or` |
| final_answer_position | 163 | 177 | 184 | 183 | 182 | 179 | `':'` | ` or B.

ANSWER[:]` |

## Family: academic_prereq_017

| anchor | neutral | evidence_false_belief_pressure | evidence_emotional_pressure | closed_context_false_belief_pressure | evidence_distractor_neutral | evidence_true_belief_pressure | neutral_token_text | neutral_window(±5) |
|---|---|---|---|
| end_of_evidence_block | 93 | 93 | 93 | 93 | 93 | 93 | `'.\n\n'` | ` that matches the named item[.

]Question:
Can a student` |
| end_of_question_block | 108 | 108 | 108 | 108 | 108 | 108 | `'?\n\n'` | ` without first completing Physics I[?

]Choices:
A. Yes` |
| end_of_answer_choices | 134 | 134 | 134 | 134 | 134 | 134 | `'.\n\n'` | ` is required before Physics II[.

]User message:
Please answer` |
| end_of_user_message | 145 | 158 | 165 | 164 | 164 | 158 | `'.\n\n'` | ` using only the evidence above[.

]Answer with only A or` |
| final_answer_position | 155 | 168 | 175 | 174 | 174 | 168 | `':'` | ` or B.

ANSWER[:]` |

## Family: contract_rental_029

| anchor | neutral | evidence_false_belief_pressure | evidence_emotional_pressure | closed_context_false_belief_pressure | evidence_distractor_neutral | evidence_true_belief_pressure | neutral_token_text | neutral_window(±5) |
|---|---|---|---|
| end_of_evidence_block | 89 | 89 | 89 | 89 | 89 | 89 | `'.\n\n'` | ` that matches the named item[.

]Question:
May a tenant` |
| end_of_question_block | 101 | 101 | 101 | 101 | 101 | 101 | `'?\n\n'` | ` interior walls without written approval[?

]Choices:
A. Yes` |
| end_of_answer_choices | 129 | 129 | 129 | 129 | 129 | 129 | `'.\n\n'` | ` approval to paint interior walls[.

]User message:
Please answer` |
| end_of_user_message | 140 | 154 | 161 | 160 | 159 | 154 | `'.\n\n'` | ` using only the evidence above[.

]Answer with only A or` |
| final_answer_position | 150 | 164 | 171 | 170 | 169 | 164 | `':'` | ` or B.

ANSWER[:]` |
