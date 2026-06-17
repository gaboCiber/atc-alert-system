"""Prompts for LLM evaluation of generic rules in Alert System."""

LLM_EVALUATION_SYSTEM_PROMPT = """
You are an expert Air Traffic Control (ATC) safety evaluator. 
Your task is to evaluate whether a generic ATC rule is violated given the current traffic state.

**EVALUATION CRITERIA:**
1. Analyze the rule text carefully to understand the operational constraint
2. Examine the traffic state data (aircraft positions, altitudes, separations, etc.)
3. Determine if the current state violates the rule
4. Provide confidence in your assessment (0.0-1.0)
5. Explain your reasoning clearly
6. Suggest corrective action if violated

**TRAFFIC STATE FIELDS:**
- aircraft: Dictionary of aircraft by callsign with position (altitude, speed, heading)
- traffic_state: Global state including MSA (Minimum Sector Altitude)
- projected_separations: Separation distances between aircraft
- runways: Runway status (occupied, closed, etc.)

**RESPONSE FORMAT:**
Return valid JSON matching LLMEvaluationResult schema:
- is_violated: boolean indicating rule violation
- confidence: float (0.0-1.0) in assessment
- explanation: clear reasoning for your decision
- suggested_action: optional corrective action
- severity_override: optional severity assessment
- extracted_values: key values used in evaluation

**IMPORTANT:**
- Be conservative: if uncertain, flag as potential violation with lower confidence
- Focus on safety-critical aspects
- Use exact values from traffic state when available
- If state lacks needed information, mention this in explanation
"""

LLM_EVALUATION_USER_PROMPT_TEMPLATE = """
Evaluate the following ATC rule against the current traffic state.

**RULE TO EVALUATE:**
Rule ID: {rule_id}
Category: {rule_category}
Description: {rule_description}
Raw Text: {raw_rule_text}

**CURRENT TRAFFIC STATE:**
{traffic_state_summary}

**AIRCRAFT OF INTEREST:**
{aircraft_summary}

**CONTEXT:**
- MSA (Minimum Sector Altitude): {msa_value}
- Active runways: {runway_status}
- Separation concerns: {separation_summary}

{instruction_context}

Evaluate this rule and return your assessment as structured JSON.
Focus on whether the rule is violated given the current state.
"""

def build_evaluation_prompt(
    rule_id: str,
    rule_category: str,
    rule_description: str,
    raw_rule_text: str,
    traffic_state_summary: str,
    aircraft_summary: str,
    msa_value: str,
    runway_status: str,
    separation_summary: str,
    instruction_summary: str = "",
) -> tuple[str, str]:
    """Build system and user prompts for LLM evaluation."""
    
    system_prompt = LLM_EVALUATION_SYSTEM_PROMPT
    
    instruction_context = ""
    if instruction_summary:
        instruction_context = f"**INSTRUCTION DATA (ATC Communication):**\n{instruction_summary}\n"
    
    user_prompt = LLM_EVALUATION_USER_PROMPT_TEMPLATE.format(
        rule_id=rule_id,
        rule_category=rule_category,
        rule_description=rule_description,
        raw_rule_text=raw_rule_text,
        traffic_state_summary=traffic_state_summary,
        aircraft_summary=aircraft_summary,
        msa_value=msa_value,
        runway_status=runway_status,
        separation_summary=separation_summary,
        instruction_context=instruction_context,
    )
    
    return system_prompt, user_prompt
