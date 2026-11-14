import json
import unittest
from unittest.mock import patch

from src.models.schemas import EvidenceExcerpt, Source, SourceSummary
from src.research.brief_generator import (
    BriefGenerationError,
    _parse_angles,
    _parse_brief,
    _source_context,
    _generate_angles,
    generate_research_brief,
)
from src.research.ingest import parse_sources
from src.research.source_summarizer import _parse_json_object, summarize_source


class ResearchPipelineTests(unittest.TestCase):
    def _synthesis_payload(self):
        return {
            "executive_summary": "Adoption is expanding while readiness varies.",
            "what_happened": "The sources report adoption, readiness, and governance findings.",
            "most_significant_finding": (
                "Together, the sources show that adoption is moving faster than readiness."
            ),
            "why_it_matters": "Operators need to plan deployment and oversight together.",
            "who_should_care": ["Founders", "Operators"],
            "source_comparison": ["The sources complement one another."],
            "analysis": ["The combined evidence indicates a readiness gap."],
            "implications": [
                "Measure readiness before scaling.",
                "Assign clear ownership for oversight.",
            ],
            "speculation": [],
            "open_questions": ["Which controls reduce incidents?", "How durable is adoption?"],
        }

    def _angle(self, title, evidence_id="source-1-evidence-1", excerpt=None):
        return {
            "title": title,
            "audience": "AI operators",
            "central_argument": f"{title} has a distinct editorial argument.",
            "why_interesting": "It connects the evidence to a practical decision.",
            "rationale": "The evidence supports a distinct editorial hook.",
            "reporting_needed": "Interview practitioners about the reported result.",
            "supporting_evidence_ids": [evidence_id],
        }

    def _angle_summary(self):
        return [
            SourceSummary(
                "source-1",
                "AI coding study",
                "https://example.com/coding",
                ["The measured result was reported."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "AI coding study",
                        "https://example.com/coding",
                        "The measured result was reported.",
                        "The measured result was reported.",
                    )
                ],
            )
        ]

    def test_two_angles_with_exact_evidence_are_accepted(self):
        payload = {
            "angle_details": [
                self._angle("Speed versus quality"),
                self._angle("The review bottleneck"),
            ]
        }
        angles, stats = _parse_angles(payload, self._angle_summary())
        self.assertEqual(len(angles), 2)
        self.assertEqual(stats["matched_evidence"], 2)

    @patch("src.research.brief_generator.chat_json")
    def test_single_angle_is_a_valid_primary_recommendation(self, chat):
        chat.return_value = json.dumps({"angle_details": [self._angle("Primary angle")]})
        angles, limitation = _generate_angles(self._angle_summary())
        self.assertEqual([angle.title for angle in angles], ["Primary angle"])
        self.assertIn("one valid", limitation.lower())

    @patch("src.research.brief_generator.chat_json")
    def test_angle_generation_failure_after_one_retry_is_non_fatal(self, chat):
        chat.side_effect = [
            json.dumps({"angle_details": []}),
            json.dumps({"angle_details": [{"title": "Malformed"}]}),
        ]
        angles, limitation = _generate_angles(self._angle_summary())
        self.assertEqual(angles, [])
        self.assertTrue(limitation)
        self.assertEqual(chat.call_count, 2)

    @patch("src.research.brief_generator.chat_json")
    def test_research_brief_survives_optional_angle_failure(self, chat):
        chat.side_effect = [
            json.dumps(
                {
                    "executive_summary": "A measured development was reported.",
                    "what_happened": "The source reported a measured result.",
                    "most_significant_finding": "The result is meaningful in context.",
                    "why_it_matters": "Operators should understand the evidence.",
                    "who_should_care": ["Operators"],
                    "source_comparison": ["The source provides one dimension."],
                    "verified_facts": ["The measured result was reported."],
                    "analysis": ["The result warrants further evaluation."],
                    "implications": ["Measure the result before scaling."],
                }
            ),
            json.dumps({"angle_details": []}),
            json.dumps({"angle_details": []}),
        ]
        brief = generate_research_brief(self._angle_summary())
        self.assertEqual(brief.executive_summary, "A measured development was reported.")
        self.assertEqual(brief.angle_details, [])
        self.assertEqual(brief.recommended_angle, "")
        self.assertIn("could not", brief.newsletter_angle_limitation.lower())

    @patch("src.research.brief_generator.chat_json")
    def test_empty_synthesis_is_retried_with_simpler_prompt(self, chat):
        empty = json.dumps({})
        chat.side_effect = [
            empty,
            json.dumps(self._synthesis_payload()),
            json.dumps({"angle_details": []}),
            json.dumps({"angle_details": []}),
        ]
        brief = generate_research_brief(self._angle_summary())
        self.assertEqual(
            brief.executive_summary,
            "Adoption is expanding while readiness varies.",
        )
        self.assertEqual(chat.call_count, 4)
        retry_prompt = chat.call_args_list[1].args[0][1]["content"]
        self.assertIn("Summarize what each source reports", retry_prompt)
        self.assertIn("provide exactly two practical", retry_prompt)

    @patch("src.research.brief_generator.chat_json")
    def test_two_empty_synthesis_responses_raise_clear_error(self, chat):
        chat.side_effect = [json.dumps({}), json.dumps({})]
        with self.assertRaisesRegex(
            BriefGenerationError,
            "missing required editorial fields.*retry also returned",
        ):
            generate_research_brief(self._angle_summary())
        self.assertEqual(chat.call_count, 2)

    @patch("src.research.brief_generator.chat_json")
    def test_angle_retry_uses_full_payload_when_model_ignores_compact_shape(self, chat):
        first = {
            "executive_summary": "Summary",
            "what_happened": "The sources report a measured result.",
            "most_significant_finding": "The combined evidence reveals a trade-off.",
            "why_it_matters": "The result affects decisions.",
            "who_should_care": ["Operators"],
            "source_comparison": ["The sources examine related evidence."],
            "analysis": ["The evidence should be evaluated together."],
            "implications": ["Measure the trade-off before scaling."],
            "open_questions": ["How durable is the result?"],
            "angle_details": [
                {
                    "title": "First angle",
                    "audience": "Operators",
                    "central_argument": "Evaluate the result.",
                    "why_interesting": "It changes a decision.",
                    "rationale": "It is evidence-grounded.",
                    "reporting_needed": "Interview operators.",
                },
                {
                    "title": "Second angle",
                    "audience": "Builders",
                    "central_argument": "Measure the result.",
                    "why_interesting": "It reveals a trade-off.",
                    "rationale": "It is distinct.",
                    "reporting_needed": "Review implementation data.",
                },
            ],
        }
        second = {
            **first,
            "source_comparison": ["The sources examine related evidence."],
            "angle_details": [
                {
                    **self._angle("First angle"),
                    "supporting_evidence_ids": ["source-1-evidence-1"],
                },
                {
                    **self._angle("Second angle"),
                    "supporting_evidence_ids": ["source-1-evidence-1"],
                },
            ],
        }
        chat.side_effect = [json.dumps(first), json.dumps(second)]
        brief = generate_research_brief(self._angle_summary())
        self.assertEqual(len(brief.angle_details), 2)
        self.assertEqual(chat.call_count, 2)

    def test_stable_evidence_id_accepts_shortened_or_paraphrased_excerpt(self):
        payload = {
            "angle_details": [
                self._angle("Coding speed", excerpt="Measured result"),
                self._angle("Review quality", excerpt="The result was measured"),
            ]
        }
        angles, _ = _parse_angles(payload, self._angle_summary())
        self.assertEqual(len(angles), 2)
        self.assertEqual(
            angles[0].evidence[0].excerpt,
            "The measured result was reported.",
        )

    def test_invalid_or_missing_evidence_references_are_rejected(self):
        for reference in (
            [],
            [{"evidence_id": "unknown", "source_id": "source-1", "excerpt": "invented"}],
            [{"evidence_id": "source-1-evidence-1", "source_id": "other", "excerpt": "The measured result was reported."}],
        ):
            payload = {
                "angle_details": [
                    {**self._angle("One"), "supporting_evidence_ids": reference},
                    {**self._angle("Two"), "supporting_evidence_ids": reference},
                ]
            }
            with self.assertRaises(BriefGenerationError):
                _parse_angles(payload, self._angle_summary())

    def test_duplicate_malformed_and_single_angles_are_rejected(self):
        duplicate = {
            "angle_details": [self._angle("Same hook"), self._angle(" same hook ")]
        }
        summaries = self._angle_summary()
        summaries[0].evidence.append(
            EvidenceExcerpt(
                "source-1",
                "AI coding study",
                "https://example.com/coding",
                "A second result was reported.",
                "A second result was reported.",
            )
        )
        angles, stats = _parse_angles(duplicate, summaries)
        self.assertEqual(len(angles), 1)
        self.assertEqual(stats["duplicate_titles"], 1)

        malformed = {
            "angle_details": [self._angle("Valid"), {"title": "Missing fields"}]
        }
        angles, stats = _parse_angles(malformed, self._angle_summary())
        self.assertEqual(len(angles), 1)
        self.assertEqual(stats["missing_fields"], 1)

        single = {"angle_details": [self._angle("Only one")]}
        angles, _ = _parse_angles(single, self._angle_summary())
        self.assertEqual(len(angles), 1)

    @patch("src.research.brief_generator.chat_json")
    def test_malformed_synthesis_json_is_repaired_once(self, chat):
        chat.side_effect = [
            '{"executive_summary": "incomplete"',
            json.dumps(
                {
                    "executive_summary": "Two sources describe the development.",
                    "what_happened": "The sources report different findings.",
                    "why_it_matters": "The distinction affects decisions.",
                    "who_should_care": ["Operators"],
                    "verified_facts": ["A measured result was reported."],
                    "analysis": ["The sources are complementary."],
                    "source_comparison": ["They examine different dimensions."],
                    "most_significant_finding": "The evidence is complementary.",
                    "implications": ["Measure the outcomes separately."],
                    "angle_details": [],
                    "evidence_records": [],
                }
            ),
            json.dumps(
                {
                    "angle_details": [
                        {
                            "title": "Angle one",
                            **self._angle("Angle one"),
                        },
                        {
                            "title": "Angle two",
                            **self._angle("Angle two"),
                        },
                    ],
                }
            ),
        ]
        summaries = [
            SourceSummary(
                "source-1",
                "Source",
                "https://example.com/source",
                ["A measured result was reported."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "Source",
                        "https://example.com/source",
                        "A measured result was reported.",
                        "A measured result was reported.",
                    )
                ],
            )
        ]
        brief = generate_research_brief(summaries)
        self.assertEqual(brief.what_happened, "The sources report different findings.")
        self.assertEqual(chat.call_count, 3)

    def test_short_pasted_source_is_not_usable(self):
        sources = parse_sources(
            urls_text="https://example.com/one\nhttps://example.com/two",
            pasted_text="Only a title",
        )
        pasted = sources[-1]
        self.assertEqual(pasted.status, "failed")
        self.assertEqual(pasted.word_count, 3)

    def test_json_parser_accepts_fenced_or_wrapped_object(self):
        payload = _parse_json_object('```json\n{"verified_facts": []}\n```')
        self.assertEqual(payload, {"verified_facts": []})
        payload = _parse_json_object('Here is the result:\n{"verified_facts": []}')
        self.assertEqual(payload, {"verified_facts": []})

    @patch("src.research.source_summarizer.time.sleep")
    @patch("src.research.source_summarizer.chat_json")
    def test_invalid_model_json_is_retried(self, chat, _sleep):
        chat.side_effect = [
            "```json\n{\"verified_facts\": [\n",
            json.dumps(
                {
                    "verified_facts": ["The study tested 100 users."],
                    "important_context": [],
                    "caveats": [],
                    "claims_to_verify": [],
                    "evidence": [],
                }
            ),
        ]
        source = Source(
            "article",
            url="https://example.com/article",
            title="Example Article",
            content="The study tested 100 users.",
            source_type="url",
        )
        summary = summarize_source(source, "source-invalid-json")
        self.assertEqual(summary.verified_facts, ["The study tested 100 users."])
        self.assertEqual(chat.call_count, 2)
        self.assertIn("only source-supported evidence", chat.call_args_list[1].args[0][1]["content"])

    @patch("src.research.source_summarizer.chat_json")
    def test_evidence_claims_are_promoted_when_facts_are_empty(self, chat):
        chat.return_value = json.dumps(
            {
                "verified_facts": [],
                "important_context": [],
                "caveats": [],
                "claims_to_verify": [],
                "evidence": [
                    {
                        "claim": "Nearly half of organizations reported an incident.",
                        "excerpt": "Nearly half of organizations reported an incident.",
                        "category": "source_reported_claim",
                    }
                ],
            }
        )
        source = Source(
            "article",
            url="https://example.com/security",
            title="Security survey",
            content="Nearly half of organizations reported an incident.",
            source_type="url",
        )
        summary = summarize_source(source, "source-evidence-only")
        self.assertEqual(
            summary.verified_facts,
            [],
        )
        self.assertEqual(
            summary.evidence[0].category,
            "source_reported_claim",
        )
        self.assertEqual(summary.findings[0].claim, summary.evidence[0].claim)

    def test_summary_facts_are_kept_when_evidence_is_empty(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Coding study",
                "https://example.com/coding",
                ["The study measured completion time for coding tasks."],
                [],
                [],
                [],
                [],
            )
        ]
        payload = {
            "angle_details": [
                {"title": "Coding speed", "central_argument": "Task time changed."},
                {"title": "Review cost", "central_argument": "Quality needs review."},
            ]
        }
        brief = _parse_brief(payload, summaries)
        self.assertIn("Coding study", brief.verified_facts[0])
        self.assertIn("Source-attributed claim (excerpt not matched)", brief.verified_facts[0])

    def test_synthesis_context_preserves_source_metadata(self):
        context = _source_context(
            [
                SourceSummary(
                    "source-metadata",
                    "Survey",
                    "https://example.com/survey",
                    ["Forty percent responded."],
                    [],
                    [],
                    [],
                    [],
                    methodology="Survey",
                    sample_size="500 respondents",
                    population="Enterprise operators",
                    date="May 2026",
                )
            ]
        )
        self.assertIn("METHODOLOGY: Survey", context)
        self.assertIn("SAMPLE SIZE: 500 respondents", context)
        self.assertIn("POPULATION: Enterprise operators", context)
        self.assertIn("DATE: May 2026", context)

    def test_empty_optional_synthesis_lists_are_not_missing_sections(self):
        brief = _parse_brief(
            {
                "executive_summary": "Summary",
                "what_happened": "The sources report a development.",
                "most_significant_finding": "The combined evidence reveals a pattern.",
                "why_it_matters": "The pattern matters to operators.",
                "who_should_care": ["Operators"],
                "source_comparison": ["The sources complement one another."],
                "analysis": ["The evidence suggests a readiness gap."],
                "implications": ["Measure readiness before scaling."],
                "speculation": [],
                "open_questions": [],
            },
            self._angle_summary(),
        )
        self.assertNotIn("open_questions", brief.missing_sections)
        self.assertNotIn("speculation", brief.missing_sections)

    def test_coding_topic_does_not_trigger_agent_specific_analysis(self):
        summaries = [
            SourceSummary(
                "source-1",
                "AI coding study",
                "https://example.com/coding",
                ["AI-assisted developers completed tasks in less time."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "AI coding study",
                        "https://example.com/coding",
                        "AI-assisted developers completed tasks in less time.",
                        "AI-assisted developers completed tasks in less time.",
                    )
                ],
            ),
            SourceSummary(
                "source-2",
                "Code quality study",
                "https://example.com/quality",
                ["The study measured maintainability separately."],
                [],
                [],
                [],
                [],
            ),
        ]
        payload = {
            "analysis": ["The studies measure different outcomes."],
            "angle_details": [
                {"title": "Speed", "central_argument": "Task time matters."},
                {"title": "Quality", "central_argument": "Maintenance matters."},
            ],
        }
        brief = _parse_brief(payload, summaries)
        self.assertTrue(all("agent" not in item.lower() for item in brief.implications))

    @patch("src.research.source_summarizer.chat_json")
    def test_source_metadata_and_exact_excerpt_survive_summarization(self, chat):
        chat.return_value = json.dumps(
            {
                "verified_facts": ["The study tested 100 users."],
                "important_context": [],
                "caveats": ["The sample was limited."],
                "claims_to_verify": [],
                "evidence": [
                    {
                        "claim": "The study tested 100 users.",
                        "excerpt": "The study tested 100 users.",
                        "category": "verified_fact",
                    }
                ],
            }
        )
        source = Source(
            "article",
            url="https://example.com/article",
            title="Example Article",
            content="The study tested 100 users.\n\nThe sample was limited.",
            source_type="url",
        )
        summary = summarize_source(source, "source-7")
        self.assertEqual(summary.source_id, "source-7")
        self.assertEqual(summary.title, "Example Article")
        self.assertEqual(summary.url, "https://example.com/article")
        self.assertEqual(summary.evidence[0].source_id, "source-7")
        self.assertEqual(summary.evidence[0].excerpt, "The study tested 100 users.")

    @patch("src.research.source_summarizer.chat_json")
    def test_source_stage_preserves_methodology_and_implication(self, chat):
        chat.return_value = json.dumps(
            {
                "topic": "Enterprise agent security",
                "central_argument": "Agent deployment creates governance risk.",
                "methodology": "Survey of security leaders",
                "sample_size": "200 respondents",
                "population": "Enterprise security teams",
                "date": "April 2026",
                "verified_facts": ["Agents exceeded intended permissions."],
                "important_context": [],
                "caveats": ["Self-reported survey."],
                "claims_to_verify": [],
                "implications": ["Teams need centralized controls."],
                "evidence": [
                    {
                        "claim": "Agents exceeded intended permissions.",
                        "excerpt": "Agents exceeded intended permissions.",
                        "category": "verified_fact",
                    }
                ],
            }
        )
        source = Source(
            "article",
            url="https://example.com/security",
            title="Security report",
            content="Agents exceeded intended permissions.",
            source_type="url",
        )
        summary = summarize_source(source, "source-structured")
        self.assertEqual(summary.methodology, "Survey of security leaders")
        self.assertEqual(summary.sample_size, "200 respondents")
        self.assertEqual(summary.population, "Enterprise security teams")
        self.assertEqual(summary.implications, ["Teams need centralized controls."])

    def test_brief_parser_replaces_source_ids_and_deduplicates_angles(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Maintainability Study",
                "https://example.com/maintainability",
                ["The study measured maintainability."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "Maintainability Study",
                        "https://example.com/maintainability",
                        "The study measured maintainability.",
                        "The study measured maintainability.",
                    )
                ],
            )
        ]
        payload = {
            "what_happened": "See source-1.",
            "verified_facts": ["source-1 measured maintainability, not productivity."],
            "source_comparison": ["source-1 does not establish productivity gains."],
            "evidence_records": [
                {
                    "source_id": "source-1",
                    "excerpt": "The study measured maintainability.",
                    "claim": "The study measured maintainability.",
                }
            ],
            "angle_details": [
                {
                    "title": "The metric problem",
                    "audience": "Engineering leaders",
                    "central_argument": "Different metrics answer different questions.",
                    "why_interesting": "It challenges broad productivity claims.",
                    "rationale": "Strong evidence boundary.",
                },
                {
                    "title": "The metric problem",
                    "audience": "Developers",
                    "central_argument": "Do not confuse maintainability with speed.",
                    "why_interesting": "It is practical.",
                    "rationale": "Clear distinction.",
                },
            ],
        }
        brief = _parse_brief(payload, summaries)
        self.assertEqual(brief.angle_details, [])
        self.assertEqual(brief.recommended_angle, "")

    def test_brief_parser_does_not_depend_on_optional_angles(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Maintainability Study",
                "https://example.com/maintainability",
                ["The study measured maintainability."],
                [],
                [],
                [],
                [],
            )
        ]
        payload = {
            "what_happened": "The sources measured different outcomes.",
            "angle_details": [
                {
                    "title": "A useful source angle",
                    "audience": "Editors",
                    "central_argument": "Compare the metrics.",
                }
            ],
        }
        brief = _parse_brief(payload, summaries)
        self.assertEqual(brief.angle_details, [])
        self.assertEqual(brief.recommended_angle, "")

    def test_brief_parser_reports_missing_editorial_sections(self):
        summaries = [
            SourceSummary(
                "source-1",
                "First source",
                "https://example.com/first",
                ["First source reported a measured result."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "First source",
                        "https://example.com/first",
                        "Measured result",
                        "First source reported a measured result.",
                    )
                ],
            ),
            SourceSummary(
                "source-2",
                "Second source",
                "https://example.com/second",
                ["Second source examined a different outcome."],
                [],
                [],
                [],
                [],
            ),
        ]
        payload = {
            "angle_details": [
                {"title": "Angle one", "central_argument": "Compare outcomes."},
                {"title": "Angle two", "central_argument": "Measure next steps."},
            ]
        }
        brief = _parse_brief(payload, summaries)
        self.assertEqual(brief.what_happened, "")
        self.assertIn("what_happened", brief.missing_sections)
        self.assertIn("why_it_matters", brief.missing_sections)
        self.assertTrue(brief.verified_facts)
        self.assertTrue(brief.evidence_records)

    def test_three_source_synthesis_preserves_cross_source_interpretation(self):
        summaries = [
            SourceSummary(
                f"source-{index}",
                f"Source {index}",
                f"https://example.com/{index}",
                [f"Direct fact {index}."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        f"source-{index}",
                        f"Source {index}",
                        f"https://example.com/{index}",
                        f"Direct fact {index}.",
                        f"Direct fact {index}.",
                        "verified_fact",
                    )
                ],
            )
            for index in range(1, 4)
        ]
        brief = _parse_brief(
            {
                "executive_summary": "Adoption is expanding while readiness differs.",
                "what_happened": "The sources measure adoption, readiness, and governance.",
                "most_significant_finding": (
                    "Together, the sources show that adoption and organizational readiness "
                    "are moving at different speeds."
                ),
                "why_it_matters": "Operators must plan for both usage and controls.",
                "who_should_care": ["Operators"],
                "source_comparison": ["The sources are complementary."],
                "analysis": ["The combined evidence reveals a readiness gap."],
                "implications": ["Measure deployment readiness before scaling."],
                "open_questions": ["Which controls reduce incidents?"],
            },
            summaries,
        )
        self.assertIn("different speeds", brief.most_significant_finding)
        self.assertEqual(len(brief.evidence_records), 3)

    def test_direct_and_source_reported_claims_remain_distinguishable(self):
        summary = SourceSummary(
            "source-claims",
            "Claims report",
            "https://example.com/claims",
            ["The report was published in 2026."],
            [],
            [],
            [],
            [
                EvidenceExcerpt(
                    "source-claims",
                    "Claims report",
                    "https://example.com/claims",
                    "The report was published in 2026.",
                    "The report was published in 2026.",
                    "verified_fact",
                ),
                EvidenceExcerpt(
                    "source-claims",
                    "Claims report",
                    "https://example.com/claims",
                    "Users reported a 40% improvement.",
                    "Users reported a 40% improvement.",
                    "source_reported_claim",
                ),
            ],
        )
        brief = _parse_brief({"analysis": ["The result needs context."]}, [summary])
        self.assertTrue(any("Directly stated in source" in fact for fact in brief.verified_facts))
        self.assertTrue(any("Source-reported claim" in fact for fact in brief.verified_facts))
        self.assertIn("40%", brief.verified_facts[-1])
        self.assertEqual(brief.recommended_angle, "")

    def test_empty_summary_uses_attributed_source_facts(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Security survey",
                "https://example.com/security",
                ["Nearly half reported an incident."],
                [],
                [],
                [],
                [],
            ),
            SourceSummary(
                "source-2",
                "Readiness survey",
                "https://example.com/readiness",
                ["Only 15% reported scaled adoption."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-2",
                        "Readiness report",
                        "https://example.com/readiness",
                        "Only 15% reported scaled adoption.",
                        "Only 15% reported scaled adoption.",
                    )
                ],
            ),
        ]
        payload = {
            "angle_details": [
                {"title": "Evidence", "central_argument": "Compare the findings."},
                {"title": "Action", "central_argument": "Measure the gaps."},
            ]
        }
        brief = _parse_brief(payload, summaries)
        self.assertEqual(brief.what_happened, "")
        self.assertIn("what_happened", brief.missing_sections)
        self.assertIn("Nearly half reported an incident.", brief.verified_facts[0])

    def test_duplicate_what_happened_and_finding_are_separated(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Source",
                "https://example.com/source",
                ["A measured result was reported."],
                [],
                [],
                [],
                [],
            )
        ]
        payload = {
            "what_happened": "The sources report a measured result.",
            "most_significant_finding": "The sources report a measured result.",
            "analysis": ["The result changes how operators should evaluate adoption."],
            "angle_details": [
                {"title": "Angle one", "central_argument": "Use the result."},
                {"title": "Angle two", "central_argument": "Question the result."},
            ],
        }
        brief = _parse_brief(payload, summaries)
        self.assertNotEqual(brief.what_happened, brief.most_significant_finding)

    def test_brief_preserves_complementary_source_findings(self):
        summaries = [
            SourceSummary(
                "source-1",
                "Security report",
                "https://example.com/security",
                ["Agents exceeded intended permissions."],
                [],
                ["Self-reported survey."],
                [],
                [
                    EvidenceExcerpt(
                        "source-1",
                        "Security report",
                        "https://example.com/security",
                        "Agents exceeded intended permissions.",
                        "Agents exceeded intended permissions.",
                    )
                ],
                topic="Agent security",
                methodology="Survey",
                population="Security teams",
            ),
            SourceSummary(
                "source-2",
                "Readiness report",
                "https://example.com/readiness",
                ["Only 15% reported scaled adoption."],
                [],
                [],
                [],
                [
                    EvidenceExcerpt(
                        "source-2",
                        "Readiness report",
                        "https://example.com/readiness",
                        "Only 15% reported scaled adoption.",
                        "Only 15% reported scaled adoption.",
                    )
                ],
                topic="Organizational readiness",
            ),
        ]
        payload = {
            "what_happened": "Deployment and governance are being measured separately.",
            "source_comparison": [
                "The sources are complementary: one measures security risk and one measures readiness."
            ],
            "analysis": ["Deployment momentum does not establish governance readiness."],
            "angle_details": [
                {"title": "Govern the rush", "central_argument": "Adoption needs controls."},
                {"title": "Readiness is the bottleneck", "central_argument": "Scale requires preparation."},
            ],
        }
        brief = _parse_brief(payload, summaries)
        self.assertEqual(len(brief.source_findings), 2)
        self.assertEqual(brief.source_findings[0].source_id, "source-1")
        self.assertIn("complementary", brief.source_comparison[0])


if __name__ == "__main__":
    unittest.main()
