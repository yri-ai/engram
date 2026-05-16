"""Tests for Ginnie Mae LoanPerf parser."""

from datetime import date
from io import BytesIO

from engram.models.track_b import DelinquencyBucket
from engram.services.track_b_ginnie_parser import parse_loanperf_records, parse_loanperf_line


SAMPLE_HEADER = b"HH|202512|Q|20260113|\n"
SAMPLE_RECORDS = (
    b"LP|36202CK51|002116|M|SF|19951101|1023374917|3871|F|1||19951201|20251101"
    b"|7.875|73000||301.28|360|359|1|0|0||94.4|||730||N|2.25|0.5|1||1|NC|||N||202510"
    b"||||||||||||||||||000000000000000000000000000000000000000000000000|\n"
    b"LP|36202CLR2|002136|M|SF|19951201|1023370953|3871|V|2||19960101|20251201"
    b"|7.875|72000||384.91|360|358|2|0|1||100|||620||N|||1||1|FL|||N||202510"
    b"||||||||||||||||||000000000000000000000000000000000000000000000000|\n"
    b"LP|36202CLR2|002136|M|SF|19951201|1023313263|3871|R|||19951201|20251101"
    b"|7.5|74000||518.04|360|359|1|0|0||96.93|||728||Y|2.25|0.5|2||1|TN|||N||202510"
    b"||||||||||||||||||000000000000000000000000000000000000000000000000|\n"
)


def test_parse_single_line():
    line = (
        "LP|36202CK51|002116|M|SF|19951101|1023374917|3871|F|1||19951201|20251101"
        "|7.875|73000||301.28|360|359|1|0|0||94.4|||730||N|2.25|0.5|1||1|NC|||N||202510"
        "||||||||||||||||||000000000000000000000000000000000000000000000000|"
    )
    event = parse_loanperf_line(line)
    assert event is not None
    assert event.loan_id == "1023374917"
    assert event.bucket == DelinquencyBucket.CURRENT
    assert event.interest_rate == 7.875
    assert event.original_upb == 73000.0
    assert event.current_upb == 301.28
    assert event.credit_score == 730
    assert event.state == "NC"
    assert event.as_of == date(2025, 10, 1)


def test_parse_delinquent_record():
    line = (
        "LP|36202CLR2|002136|M|SF|19951201|1023370953|3871|V|2||19960101|20251201"
        "|7.875|72000||384.91|360|358|2|0|1||100|||620||N|||1||1|FL|||N||202510"
        "||||||||||||||||||000000000000000000000000000000000000000000000000|"
    )
    event = parse_loanperf_line(line)
    assert event is not None
    assert event.loan_id == "1023370953"
    assert event.bucket == DelinquencyBucket.D30
    assert event.state == "FL"
    assert event.credit_score == 620


def test_parse_reo_record():
    line = (
        "LP|36202CLR2|002136|M|SF|19951201|1023313263|3871|R|||19951201|20251101"
        "|7.5|74000||518.04|360|359|1|0|0||96.93|||728||Y|2.25|0.5|2||1|TN|||N||202510"
        "||||||||||||||||||000000000000000000000000000000000000000000000000|"
    )
    event = parse_loanperf_line(line)
    assert event is not None
    assert event.bucket == DelinquencyBucket.REO


def test_parse_loanperf_records_from_stream():
    data = SAMPLE_HEADER + SAMPLE_RECORDS
    stream = BytesIO(data)
    events = list(parse_loanperf_records(stream, limit=10))
    assert len(events) == 3
    assert events[0].loan_id == "1023374917"
    assert events[1].loan_id == "1023370953"
    assert events[2].loan_id == "1023313263"


def test_parse_skips_non_lp_lines():
    data = b"HH|202512|Q|20260113|\nNOT_A_RECORD\n" + SAMPLE_RECORDS
    stream = BytesIO(data)
    events = list(parse_loanperf_records(stream, limit=10))
    assert len(events) == 3


def test_parse_respects_limit():
    data = SAMPLE_HEADER + SAMPLE_RECORDS
    stream = BytesIO(data)
    events = list(parse_loanperf_records(stream, limit=2))
    assert len(events) == 2
