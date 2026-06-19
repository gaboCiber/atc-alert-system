import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import E6Config, LLMConfig
from evaluator import run_benchmark
from report import generate_report
from semantic_judge import SemanticJudge

RULE_DESCRIPTIONS: dict = {
    # ── Non-compilable: Communication quality / subjective ──
    "RULE001": (
        "Communication Quality Rule: ATC communications must be clear, concise, and unambiguous. "
        "A violation occurs if the instruction text is unclear, verbose, or ambiguous."
    ),
    "RULE002": (
        "Professional Standards Rule: All RTF communications must be conducted using the highest professional standards. "
        "A violation occurs if the instruction uses informal language or deviates from standard phraseology."
    ),
    "RULE003": (
        "Verbosity Rule: Communications should be brief and to the point. "
        "A violation occurs if the instruction contains unnecessary words or excessive detail."
    ),
    "RULE008": (
        "Noting Down Rule: Pilots must note down clearances before acting on them. "
        "Cannot be verified from traffic state alone."
    ),
    "RULE012": (
        "Crossing Confirmation Rule: When an aircraft crosses a runway, ATC must request confirmation. "
        "A violation occurs if the aircraft crosses without explicit confirmation."
    ),
    "RULE017": (
        "Conditional Execution Rule: Conditional clearances must be executed only when the condition is met. "
        "Cannot verify actual execution from traffic state."
    ),
    "RULE020": (
        "Engine Start Rule: When aircraft is at stand and ready to start, ATC must issue startup approval. "
        "A violation occurs if startup is not approved when the aircraft is ready."
    ),
    "RULE023": (
        "Startup Approval Content Rule: Startup approval must include frequency change information. "
        "A violation occurs if the instruction omits frequency details."
    ),
    "RULE037": (
        "Metro Tower Contact Rule: Aircraft at holding point C1 must establish contact with Metro Tower. "
        "Requires tracking communication state not available in TrafficState."
    ),
    "RULE045": (
        "Holding Point C1 Rule: Aircraft must identify holding point C1 by its geographical reference. "
        "Holding point C1 is not defined in TrafficState."
    ),
    "RULE051": (
        "Departure Information Rule: Departure information must not be confused with takeoff clearance. "
        "Requires subjective judgment about communication content."
    ),
    "RULE058": (
        "Correct Action Rule: Flight crew must perform the correct action following a read-back. "
        "Cannot verify physical cockpit actions from traffic state."
    ),
    "RULE059": (
        "Noting Down Clearance Rule: Pilots must note down clearances. "
        "Physical cockpit action not capturable in TrafficState."
    ),
    "RULE060": (
        "Dual Crew Listening Rule: Both crew members must listen to clearances. "
        "Requires cockpit monitoring data not in TrafficState."
    ),
    "RULE061": (
        "Doubt Detection Rule: If doubt exists regarding the intended plan, clarification must be sought. "
        "Subjective mental state not detectable from traffic state."
    ),
    "RULE077": (
        "Read-back Integrity Rule: Read-backs must accurately reflect the original instruction. "
        "Requires comparing pilot response to ATC instruction."
    ),
    "RULE081": (
        "Initial Contact Quality Rule: Initial calls must contain all required information. "
        "Requires judging completeness of communication."
    ),
    "RULE082": (
        "Omission Detection Rule: If information is omitted in initial call, a clarification call is needed. "
        "Requires tracking communication sequence."
    ),
    "RULE092": (
        "Enunciation Rule: Special care must be taken when enunciating 'zero zero'. "
        "Requires voice quality analysis not available in text."
    ),
    "RULE095": (
        "RVSM Resume Rule: When ability to resume RVSM is confirmed, pilot must say 'Ready to resume RVSM'. "
        "Requires tracking RVSM state."
    ),
    "RULE096": (
        "Non-RVSM Aircraft Rule: Must identify if aircraft is non-RVSM approved. "
        "Aircraft certification status not in TrafficState."
    ),
    "RULE100": (
        "RA Detection Rule: Resolution Advisory (RA) from TCAS must be detected. "
        "TCAS data not available in TrafficState."
    ),
    "RULE101": (
        "RA Responsibility Rule: When pilot reports RA, controller responsibility changes. "
        "Requires tracking RA state."
    ),
    "RULE102": (
        "RA Maneuver Rule: Aircraft performing RA-induced maneuver must be identified. "
        "TCAS RA data not in TrafficState."
    ),
    "RULE103": (
        "RA Departure Rule: RA causing departure from ATC clearance must be detected. "
        "TCAS internal state not available."
    ),
    "RULE104": (
        "RA Conflict Rule: RA conflict between aircraft must be detected. "
        "TCAS conflict data not in TrafficState."
    ),
    "RULE105": (
        "RA Resume Rule: After RA, pilot must report resuming assigned clearance. "
        "Requires tracking RA resolution state."
    ),
    "RULE106": (
        "ACAS RA Rule: ACAS Resolution Advisory must be detected. "
        "ACAS data not available in TrafficState."
    ),
    "RULE107": (
        "ACAS RA Response Rule: Response to ACAS RA must be verified. "
        "ACAS state not in TrafficState."
    ),
    "RULE108": (
        "TMA Conditional Rule: Conditional clearances in TMA require special handling. "
        "TMA location detection requires geographical data."
    ),
    "RULE109": (
        "Execute With Care Rule: Conditional clearances must be executed with care. "
        "Subjective quality of performance."
    ),
    "RULE111": (
        "Doubt Ambiguity Rule: When doubt or ambiguity exists, it must be resolved. "
        "Subjective internal state not detectable."
    ),
    "RULE112": (
        "Note Down Clearance Rule: Flight crew must note down clearances. "
        "Physical cockpit action."
    ),
    "RULE115": (
        "Voice Tone Rule: Urgent tone must be detected in communications. "
        "Audio quality analysis not available in text."
    ),
    "RULE118": (
        "Simultaneous Transmission Rule: Simultaneous transmissions must be detected. "
        "Requires timing/overlap analysis."
    ),
    "RULE123": (
        "ILS Approach Rule: Cleared ILS approach must include runway specification. "
        "Requires tracking approach clearance content."
    ),
    "RULE124": (
        "Descent After Approach Rule: Descent clearance after approach must be verified. "
        "Requires tracking altitude vs approach state."
    ),
    "RULE126": (
        "Maintain Until Glide Path Rule: Must maintain altitude until intercepting glide path. "
        "Requires tracking glide path interception."
    ),
    "RULE127": (
        "Report Established Rule: Must report established on glide path when instructed. "
        "Requires tracking glide path position."
    ),
    "RULE128": (
        "Initial Contact Info Rule: Initial contact must include required information. "
        "Requires judging completeness."
    ),
    "RULE133": (
        "Clearance Content Rule: Clearances must contain all required elements. "
        "Requires subjective judgment about completeness."
    ),
    "RULE134": (
        "ILS Altitude Rule: ILS approach clearance must include altitude assignment. "
        "Requires tracking approach and altitude state."
    ),
    "RULE135": (
        "Busy RTF Rule: In busy RTF situations, special procedures apply. "
        "Subjective busy state not in TrafficState."
    ),
    "RULE136": (
        "Safe Landing Availability Rule: Aircraft must be available in good time for safe landing. "
        "Subjective temporal prediction."
    ),
    "RULE142": (
        "Missed Approach Rule: During missed approach, specific procedures must be followed. "
        "Requires tracking approach state."
    ),
    "RULE143": (
        "Brief Transmission Rule: Transmissions must be brief and minimum. "
        "Subjective quality assessment."
    ),
    "RULE147": (
        "Safe Conduct Doubt Rule: If doubt exists about safe conduct, action must be taken. "
        "Subjective internal state."
    ),
    "RULE148": (
        "Emergency Procedures Rule: Emergency procedures must follow specific format. "
        "Requires subjective judgment."
    ),
    "RULE151": (
        "Emergency Initial Call Rule: Emergency situations require specific initial call format. "
        "Requires tracking emergency state."
    ),
    "RULE153": (
        "Communication Sequence Rule: Communication sequences must follow proper order. "
        "Classification error - requires subjective assessment."
    ),
    "RULE156": (
        "Phraseology Compliance Rule: Phraseology must comply with ICAO standards. "
        "Classification error - requires subjective assessment."
    ),
    "RULE157": (
        "Read-back Completeness Rule: Read-backs must be complete. "
        "Classification error - requires subjective assessment."
    ),
    "RULE158": (
        "Instruction Clarity Rule: Instructions must be clear and understandable. "
        "Classification error - requires subjective assessment."
    ),
    "RULE159": (
        "Communication Timing Rule: Communications must be properly timed. "
        "Classification error - requires subjective assessment."
    ),
    # ── Compilable: Read-back rules ──
    "RULE011": (
        "Read-back Rule: When a clearance is issued, the pilot must read it back with specific elements. "
        "A violation occurs if the read-back is missing or incorrect."
    ),
    "RULE015": (
        "Taxi Read-back Rule: Taxi and runway crossing instructions must be fully read back. "
        "A violation occurs if the pilot does not read back the full taxi instruction."
    ),
    "RULE016": (
        "Conditional Taxi Read-back Rule: Conditional taxi clearances must be read back in full. "
        "A violation occurs if the conditional clearance is not read back completely."
    ),
    "RULE024": (
        "Startup Read-back Rule: Start-up approval and frequency change must be read back. "
        "A violation occurs if the read-back omits startup or frequency details."
    ),
    "RULE030": (
        "Frequency Change Read-back Rule: Frequency change instructions must be read back. "
        "A violation occurs if the pilot does not read back the new frequency."
    ),
    "RULE034": (
        "Hold Read-back Rule: HOLD/HOLD POSITION instructions must be read back. "
        "A violation occurs if the hold instruction is not read back."
    ),
    "RULE039": (
        "Line-up Read-back Rule: Line-up instructions must be read back. "
        "A violation occurs if the line-up clearance is not read back."
    ),
    "RULE054": (
        "Hold At Read-back Rule: 'Hold at' instructions must be read back. "
        "A violation occurs if the hold-at instruction is not read back."
    ),
    "RULE062": (
        "Safety Message Read-back Rule: Safety-related messages must be read back. "
        "A violation occurs if the safety message is not acknowledged."
    ),
    "RULE063": (
        "Taxi Instruction Read-back Rule: Taxi instructions must be read back in full. "
        "A violation occurs if the taxi instruction is not read back."
    ),
    "RULE064": (
        "Level Instruction Read-back Rule: Level/altitude instructions must be read back. "
        "A violation occurs if the level instruction is not read back."
    ),
    "RULE065": (
        "Heading Instruction Read-back Rule: Heading instructions must be read back. "
        "A violation occurs if the heading instruction is not read back."
    ),
    "RULE066": (
        "Speed Instruction Read-back Rule: Speed instructions must be read back. "
        "A violation occurs if the speed instruction is not read back."
    ),
    "RULE069": (
        "Route Read-back Rule: Route/airway clearances must be read back. "
        "A violation occurs if the route clearance is not read back."
    ),
    "RULE071": (
        "SSR Read-back Rule: SSR operating instructions must be read back. "
        "A violation occurs if the SSR instruction is not read back."
    ),
    "RULE072": (
        "Altimeter Read-back Rule: Altimeter/QNH settings must be read back. "
        "A violation occurs if the altimeter setting is not read back."
    ),
    "RULE073": (
        "VDF Information Read-back Rule: VDF information must be read back. "
        "A violation occurs if VDF info is not acknowledged."
    ),
    "RULE074": (
        "Radar Service Read-back Rule: Radar service type information must be read back. "
        "A violation occurs if radar service type is not acknowledged."
    ),
    "RULE075": (
        "Transition Level Read-back Rule: Transition level information must be read back. "
        "A violation occurs if transition level is not acknowledged."
    ),
    "RULE078": (
        "Read-back Integrity Check Rule: Read-back must match the original instruction. "
        "A violation occurs if the read-back differs from the instruction."
    ),
    "RULE079": (
        "Closed-loop Communication Rule: Communication must follow closed-loop sequence. "
        "A violation occurs if the communication loop is not completed."
    ),
    "RULE083": (
        "First Contact Callsign Rule: First contact on new frequency must include callsign. "
        "A violation occurs if callsign is omitted in initial contact."
    ),
    "RULE084": (
        "First Contact Wake Turbulence Rule: First contact must include wake turbulence category. "
        "A violation occurs if wake turbulence category is omitted."
    ),
    "RULE085": (
        "First Contact Clearance Rule: First contact must include current clearance. "
        "A violation occurs if clearance is not reported in initial contact."
    ),
    "RULE086": (
        "First Contact Level Rule: First contact must include assigned level. "
        "A violation occurs if assigned level is not reported."
    ),
    "RULE087": (
        "First Contact Position Rule: First contact must include position information. "
        "A violation occurs if position is not reported."
    ),
    "RULE088": (
        "First Contact Assigned Clearances Rule: First contact must include all assigned clearances. "
        "A violation occurs if clearances are not fully reported."
    ),
    "RULE091": (
        "Flight Level Phraseology Rule: Flight levels must use correct phraseology. "
        "A violation occurs if 'flight level' is not used correctly."
    ),
    "RULE093": (
        "Affirm Negative Rule: Responses must use 'Affirm' or 'Negative' correctly. "
        "A violation occurs if incorrect response phraseology is used."
    ),
    "RULE094": (
        "Denial Phraseology Rule: Denials must use specific phraseology. "
        "A violation occurs if denial does not use standard phraseology."
    ),
    "RULE098": (
        "Clearance Read-back Content Rule: Clearance read-back must contain all elements. "
        "A violation occurs if read-back is incomplete."
    ),
    "RULE110": (
        "Conditional Clearance Read-back Format Rule: Conditional clearance read-back must match format. "
        "A violation occurs if read-back format differs from original."
    ),
    "RULE116": (
        "VHF Frequency Format Rule: VHF frequencies must be communicated in correct format. "
        "A violation occurs if frequency format is incorrect."
    ),
    "RULE117": (
        "Frequency Transmission Rule: Frequency transmissions must follow specific format. "
        "A violation occurs if frequency is not transmitted correctly."
    ),
    "RULE119": (
        "Affirmative Phraseology Rule: 'Affirm' must be used instead of 'yes'. "
        "A violation occurs if 'yes' is used instead of 'affirm'."
    ),
    "RULE120": (
        "Ambiguity Detection Rule: Ambiguous words must not be used in communications. "
        "A violation occurs if ambiguous terminology is detected."
    ),
    "RULE129": (
        "QNH Read-back Rule: New QNH settings must be read back. "
        "A violation occurs if QNH is not read back."
    ),
    "RULE130": (
        "Heading Altitude Speed Read-back Rule: Heading, altitude, and speed instructions must be read back. "
        "A violation occurs if any of these instructions is not read back."
    ),
    "RULE132": (
        "Report Established Rule: When instructed to report established, pilot must comply. "
        "A violation occurs if the report is not made when instructed."
    ),
    "RULE137": (
        "Landing Clearance Delay Reason Rule: When landing clearance is delayed, a reason must be given. "
        "A violation occurs if no reason is provided for delay."
    ),
    "RULE138": (
        "Continue Not Categorized Rule: 'Continue' must not be used as a clearance. "
        "A violation occurs if 'continue' is used as a clearance category."
    ),
    "RULE144": (
        "Going Around Phraseology Rule: 'Going around' must be used during missed approach. "
        "A violation occurs if incorrect phraseology is used."
    ),
    "RULE145": (
        "Go Around Instruction Phraseology Rule: Go around instructions must use correct phraseology. "
        "A violation occurs if phraseology is incorrect."
    ),
    "RULE146": (
        "Go Around Read-back Rule: Go around instructions must be read back. "
        "A violation occurs if go around instruction is not read back."
    ),
    "RULE149": (
        "Mayday Prefix Rule: Distress calls must be prefixed with 'MAYDAY'. "
        "A violation occurs if MAYDAY prefix is omitted."
    ),
    "RULE150": (
        "Pan-Pan Prefix Rule: Urgency calls must be prefixed with 'PAN-PAN'. "
        "A violation occurs if PAN-PAN prefix is omitted."
    ),
    "RULE152": (
        "Emergency Squawk Rule: Emergency aircraft must use squawk 7700. "
        "A violation occurs if emergency squawk is not set."
    ),
    "RULE154": (
        "Communication Sequence Order Rule: Communication sequences must follow proper order. "
        "A violation occurs if sequence order is incorrect."
    ),
    "RULE155": (
        "Read-back Timeliness Rule: Read-backs must be timely. "
        "A violation occurs if read-back is delayed excessively."
    ),
    # ── Compilable: Traffic state rules ──
    "RULE004": (
        "Runway Rule: No aircraft may be on a runway that is occupied by another aircraft. "
        "A violation occurs if an aircraft is within 3 NM of an occupied runway."
    ),
    "RULE005": (
        "Runway Incursion Rule: Aircraft must not enter a runway without clearance. "
        "A violation occurs if an aircraft enters an occupied runway without valid clearance."
    ),
    "RULE006": (
        "Taxi Clearance Limit Rule: Taxi clearances must include a clearance limit. "
        "A violation occurs if a taxi clearance omits the clearance limit."
    ),
    "RULE007": (
        "Clearance Limit Stop Rule: Aircraft must stop at the clearance limit. "
        "A violation occurs if the aircraft does not stop at the specified clearance limit."
    ),
    "RULE009": (
        "Altitude MSA Rule: Aircraft must maintain altitude above MSA. "
        "A violation occurs if aircraft altitude is below minimum safe altitude."
    ),
    "RULE010": (
        "Departure Sequence Rule: Departure instructions must not confuse information with clearance. "
        "A violation occurs if departure information is issued as takeoff clearance."
    ),
    "RULE013": (
        "Conditional Crossing Rule: Conditional runway crossing must meet conditions. "
        "A violation occurs if crossing instruction does not include proper conditions."
    ),
    "RULE014": (
        "Crossing Confirmation Rule: Runway crossing must be confirmed by ATC. "
        "A violation occurs if aircraft crosses runway without ATC confirmation."
    ),
    "RULE018": (
        "Conditional Clearance Identification Rule: Ambiguous conditional clearances must include identification. "
        "A violation occurs if identification (livery/colour) is missing."
    ),
    "RULE019": (
        "Departure Clearance Content Rule: Departure clearances must contain all required elements. "
        "A violation occurs if departure clearance is incomplete."
    ),
    "RULE021": (
        "Clearance Content Rule: Clearances must contain all required parameters. "
        "A violation occurs if clearance is missing required elements."
    ),
    "RULE022": (
        "Departure Clearance Format Rule: Departure clearances must follow correct format. "
        "A violation occurs if format is incorrect."
    ),
    "RULE029": (
        "Conditional Taxi Crossing Rule: Conditional taxi to cross runway must be specific. "
        "A violation occurs if conditional crossing is not specific enough."
    ),
    "RULE031": (
        "Clearance Phraseology Rule: Clearances must use correct phraseology. "
        "A violation occurs if phraseology is incorrect."
    ),
    "RULE032": (
        "Instruction Phraseology Rule: Instructions must use correct phraseology. "
        "A violation occurs if phraseology deviates from standard."
    ),
    "RULE033": (
        "Takeoff Clearance Phraseology Rule: Takeoff clearances must use correct phraseology. "
        "A violation occurs if takeoff phraseology is incorrect."
    ),
    "RULE035": (
        "Cleared Word Usage Rule: The word 'cleared' must be used in takeoff/landing clearances. "
        "A violation occurs if 'cleared' is not used."
    ),
    "RULE036": (
        "Takeoff Clearance Content Rule: Takeoff clearance must contain only takeoff instruction. "
        "A violation occurs if takeoff clearance contains additional instructions."
    ),
    "RULE038": (
        "Line-up Instruction Rule: When aircraft is at holding point and runway is ready, line-up must be issued. "
        "A violation occurs if line-up is not issued when conditions are met."
    ),
    "RULE040": (
        "Takeoff Clearance Timing Rule: Takeoff clearance must be issued at the correct time. "
        "A violation occurs if takeoff clearance is issued prematurely."
    ),
    "RULE041": (
        "Airborne Reporting Rule: Aircraft must report when airborne. "
        "A violation occurs if airborne report is not made."
    ),
    "RULE042": (
        "Approach Clearance Rule: Approach clearances must be issued correctly. "
        "A violation occurs if approach clearance is incorrect."
    ),
    "RULE043": (
        "Departure Amendment Read-back Rule: Departure amendments must be read back. "
        "A violation occurs if amendment is not read back."
    ),
    "RULE044": (
        "Departure Amendment Content Rule: Departure amendments must contain all required elements. "
        "A violation occurs if amendment is incomplete."
    ),
    "RULE046": (
        "Conditional Line-up Phraseology Rule: Conditional line-up clearances must follow correct order. "
        "A violation occurs if phraseology order is incorrect."
    ),
    "RULE055": (
        "Takeoff Roll Interrupt Rule: When takeoff roll is interrupted, hold position instruction and reason must be given. "
        "A violation occurs if reason is not provided."
    ),
    "RULE056": (
        "Takeoff Cancellation Rule: Takeoff cancellation must follow specific procedure. "
        "A violation occurs if procedure is not followed."
    ),
    "RULE057": (
        "Cancel Takeoff Reason Rule: Cancel takeoff instruction must include a reason. "
        "A violation occurs if reason is not provided."
    ),
    "RULE067": (
        "Route Clearance Content Rule: Route clearances must contain all required elements. "
        "A violation occurs if route clearance is incomplete."
    ),
    "RULE068": (
        "Approach Clearance Content Rule: Approach clearances must contain all required elements. "
        "A violation occurs if approach clearance is incomplete."
    ),
    "RULE070": (
        "Runway Holding Rule: Aircraft must hold short of runway when instructed. "
        "A violation occurs if aircraft does not hold short."
    ),
    "RULE076": (
        "Read-back Verification Rule: Read-backs must be verified by ATC. "
        "A violation occurs if read-back is not verified."
    ),
    "RULE080": (
        "Altitude Clearance Rule: Altitude clearances must be specific and correct. "
        "A violation occurs if altitude clearance is ambiguous."
    ),
    "RULE089": (
        "Affirmative Response Rule: Responses must use 'Affirm' or 'Roger' correctly. "
        "A violation occurs if response phraseology is incorrect."
    ),
    "RULE090": (
        "Heading Degrees Rule: Heading instructions must include 'degrees' when appropriate. "
        "A violation occurs if 'degrees' is omitted."
    ),
    "RULE094": (
        "Denial Phraseology Rule: Denials must use standard phraseology. "
        "A violation occurs if non-standard phraseology is used."
    ),
    "RULE100": (
        "Resolution Advisory Rule: TCAS Resolution Advisories must be properly handled. "
        "Requires TCAS data not in TrafficState."
    ),
    "RULE113": (
        "Lateral Collision Risk Rule: Lateral collision risk must be assessed and communicated. "
        "A violation occurs if collision risk is not addressed."
    ),
    "RULE114": (
        "Head-on Collision Risk Rule: Head-on collision risk must be assessed. "
        "A violation occurs if head-on risk is not addressed."
    ),
    "RULE121": (
        "Reduced Separation Request Rule: Pilots must not request reduced vortex wake separation. "
        "A violation occurs if pilot requests reduced separation."
    ),
    "RULE122": (
        "Reduced Separation Grant Rule: Controllers must not grant reduced vortex separation. "
        "A violation occurs if controller grants reduced separation."
    ),
    "RULE125": (
        "Approach Clearance Condition Rule: Approach clearances must include conditions when needed. "
        "A violation occurs if conditions are missing."
    ),
    "RULE131": (
        "Runway Occupancy Rule: Aircraft must not occupy runway when another aircraft is landing. "
        "A violation occurs if runway occupancy conflicts with landing."
    ),
    "RULE139": (
        "Final Approach Rule: Aircraft on final approach must be monitored. "
        "A violation occurs if final approach monitoring fails."
    ),
    "RULE140": (
        "Landing Clearance Content Rule: Landing clearances must include wind information. "
        "A violation occurs if wind info is missing."
    ),
    "RULE141": (
        "Landing Read-back Rule: Landing clearances must be read back. "
        "A violation occurs if landing clearance is not read back."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="E6: System-level Latency & Accuracy Benchmark")
    parser.add_argument("--base-dir", type=str, default=None, help="Override base dir")
    parser.add_argument("--ground-truth-dir", type=str, default=None, help="Override ground truth dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--compiled-rules-dir", type=str, default=None, help="Override compiled rules dir")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="LLM model for generic + judge")
    parser.add_argument("--provider", type=str, default="openai", help="Provider (openai, gemini, anthropic)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", help="LLM API base URL")
    parser.add_argument("--api-key", type=str, default="ollama", help="LLM API key")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--measure", type=int, default=15, help="Measurement iterations")
    args = parser.parse_args()

    cfg = E6Config.from_dirs(
        base_dir=args.base_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        compiled_rules_dir=args.compiled_rules_dir,
    )
    cfg.warmup_iterations = args.warmup
    cfg.measure_iterations = args.measure

    print(f"E6 System Benchmark")
    print(f"  Ground truth: {cfg.ground_truth_dir}")
    print(f"  Compiled rules: {cfg.compiled_rules_dir}")
    print(f"  LLM: {args.model} ({args.provider})")
    print(f"  Iterations: {cfg.warmup_iterations} warmup + {cfg.measure_iterations} measure")
    if args.no_judge:
        print("  Judge: disabled")
    print()

    llm_cfg = LLMConfig(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    judge = None
    if not args.no_judge:
        judge = SemanticJudge(llm_cfg)

    results = run_benchmark(cfg, llm_cfg, judge, RULE_DESCRIPTIONS)

    report_info = generate_report(results, cfg, save_figures=True)

    print(f"Report: {report_info['results']}")
    print(f"Summary: {report_info['summary']}")
    print(f"Figures: {len(report_info['figures'])} generated")

    if results.judge_scores:
        avg_judge = sum(results.judge_scores.values()) / len(results.judge_scores)
        print(f"Avg judge score: {avg_judge:.3f}")

    # Accuracy summary
    for rule_id, rm in results.accuracy.items():
        print(f"  {rule_id}: P={rm.precision:.2f} R={rm.recall:.2f} F1={rm.f1:.2f}")

    # Ranking
    if results.ranking:
        for entry in results.ranking:
            print(f"  {entry['component']}: {entry['avg_latency_ms']}ms")

    return results


if __name__ == "__main__":
    main()
