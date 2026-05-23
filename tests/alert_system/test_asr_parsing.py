"""Tests de parseo ASR: raw text -> ParsedInstruction."""

import pytest
from datetime import datetime

from Alert_System.models.instruction import InstructionType, Speaker, ParsedInstruction
from Alert_System.integration import ASRAdapter
from ASR.transcription.base import TranscriptionResult


def make_transcription(text: str) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        file_path="test.wav",
        model_name="test",
    )


class TestASRParsing:
    """Tests del parseo ATC desde raw text a ParsedInstruction."""

    @pytest.fixture
    def adapter(self):
        return ASRAdapter()

    def _adapt(self, adapter: ASRAdapter, text: str) -> ParsedInstruction:
        return adapter.adapt(make_transcription(text), Speaker.ATCO)

    # ── Altitude ──

    def test_descend_fl(self, adapter):
        p = self._adapt(adapter, "AAL123 descend to FL240")
        assert p.callsign == "AAL123"
        assert p.instruction_type == InstructionType.DESCENT
        assert p.parameters["target_altitude"] == 24000

    def test_descend_altitude(self, adapter):
        p = self._adapt(adapter, "UAL456 descend to 3000")
        assert p.callsign == "UAL456"
        assert p.instruction_type == InstructionType.DESCENT
        assert p.parameters["target_altitude"] == 3000

    def test_climb_fl(self, adapter):
        p = self._adapt(adapter, "DAL789 climb to FL350")
        assert p.callsign == "DAL789"
        assert p.instruction_type == InstructionType.CLIMB
        assert p.parameters["target_altitude"] == 35000

    def test_climb_altitude(self, adapter):
        p = self._adapt(adapter, "SWA123 climb to 12000")
        assert p.callsign == "SWA123"
        assert p.instruction_type == InstructionType.CLIMB
        assert p.parameters["target_altitude"] == 12000

    def test_maintain_fl(self, adapter):
        p = self._adapt(adapter, "AAL123 maintain FL330")
        assert p.callsign == "AAL123"
        assert p.instruction_type == InstructionType.MAINTAIN_ALTITUDE
        assert p.parameters["target_altitude"] == 33000

    def test_maintain_altitude(self, adapter):
        p = self._adapt(adapter, "UAL456 maintain 10000 feet")
        assert p.callsign == "UAL456"
        assert p.instruction_type == InstructionType.MAINTAIN_ALTITUDE
        assert p.parameters["target_altitude"] == 10000

    def test_altitude_with_ft_suffix(self, adapter):
        p = self._adapt(adapter, "DAL789 descend to 5000 feet")
        assert p.instruction_type == InstructionType.DESCENT
        assert p.parameters["target_altitude"] == 5000

    # ── Heading ──

    def test_heading_with_direction(self, adapter):
        p = self._adapt(adapter, "UAL456 turn left heading 270")
        assert p.callsign == "UAL456"
        assert p.instruction_type == InstructionType.HEADING
        assert p.parameters["heading"] == 270
        assert p.parameters["direction"] == "left"

    def test_heading_without_direction(self, adapter):
        p = self._adapt(adapter, "DAL789 fly heading 180")
        assert p.callsign == "DAL789"
        assert p.instruction_type == InstructionType.HEADING
        assert p.parameters["heading"] == 180

    def test_turn_left(self, adapter):
        p = self._adapt(adapter, "AAL123 turn left")
        assert p.instruction_type == InstructionType.HEADING
        assert p.parameters.get("direction") == "left"

    def test_turn_right(self, adapter):
        p = self._adapt(adapter, "UAL456 turn right")
        assert p.instruction_type == InstructionType.HEADING
        assert p.parameters.get("direction") == "right"

    # ── Runway / Takeoff / Landing ──

    def test_takeoff_clearance(self, adapter):
        p = self._adapt(adapter, "DAL789 cleared for takeoff runway 04L")
        assert p.callsign == "DAL789"
        assert p.instruction_type == InstructionType.TAKEOFF_CLEARANCE
        assert p.parameters["runway"] == "04L"

    def test_takeoff_compact_rwy(self, adapter):
        p = self._adapt(adapter, "UAL456 cleared for takeoff RWY34R")
        assert p.instruction_type == InstructionType.TAKEOFF_CLEARANCE
        assert p.parameters["runway"] == "34R"

    def test_landing_clearance(self, adapter):
        p = self._adapt(adapter, "AAL123 cleared to land runway 27R")
        assert p.callsign == "AAL123"
        assert p.instruction_type == InstructionType.LANDING_CLEARANCE
        assert p.parameters["runway"] == "27R"

    def test_approach_clearance(self, adapter):
        p = self._adapt(adapter, "SWA456 cleared for approach runway 27R")
        assert p.instruction_type == InstructionType.APPROACH_CLEARANCE

    # ── Speed ──

    def test_speed_direct(self, adapter):
        p = self._adapt(adapter, "UAL456 speed 280")
        assert p.callsign == "UAL456"
        assert p.instruction_type == InstructionType.SPEED
        assert p.parameters["target_speed"] == 280

    def test_reduce_speed(self, adapter):
        p = self._adapt(adapter, "AAL123 reduce speed to 250 knots")
        assert p.instruction_type == InstructionType.REDUCE_SPEED
        assert p.parameters["target_speed"] == 250

    def test_increase_speed(self, adapter):
        p = self._adapt(adapter, "DAL789 increase speed to 300")
        assert p.instruction_type == InstructionType.INCREASE_SPEED
        assert p.parameters["target_speed"] == 300

    # ── Contact ──

    def test_contact_approach(self, adapter):
        p = self._adapt(adapter, "AAL123 contact approach 118.5")
        assert p.callsign == "AAL123"
        assert p.instruction_type == InstructionType.APPROACH_CLEARANCE

    def test_contact_frequency(self, adapter):
        p = self._adapt(adapter, "UAL456 contact departure frequency 124.3")
        assert p.instruction_type == InstructionType.CONTACT

    # ── Hold ──

    def test_hold_position(self, adapter):
        p = self._adapt(adapter, "DAL789 hold short of runway 27")
        assert p.callsign == "DAL789"
        assert p.instruction_type == InstructionType.HOLD_POSITION

    # ── Edge cases ──

    def test_unknown_instruction(self, adapter):
        p = self._adapt(adapter, "something unexpected here")
        assert p.instruction_type == InstructionType.UNKNOWN

    def test_missing_callsign(self, adapter):
        p = self._adapt(adapter, "descend to FL200")
        assert p.callsign is None
        assert p.instruction_type == InstructionType.DESCENT

    def test_speaker_pilot(self, adapter):
        p = adapter.adapt(
            make_transcription("AAL123 requesting descent"),
            Speaker.PILOT,
        )
        assert p.speaker == Speaker.PILOT
        assert p.callsign == "AAL123"

    def test_callsign_variations(self, adapter):
        for raw in ["AAL123", "UAL456", "DAL789", "SWA1234", "JBU123"]:
            p = self._adapt(adapter, f"{raw} climb to FL300")
            assert p.callsign == raw, f"Failed for callsign {raw}"
