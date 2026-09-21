import pandas as pd

from bess_trading.data.entsoe import parse_timeseries


def test_parse_price_xml() -> None:
    xml = b"""<Publication_MarketDocument xmlns="urn:test"><TimeSeries><Period>
    <timeInterval><start>2024-01-01T00:00Z</start></timeInterval><resolution>PT60M</resolution>
    <Point><position>1</position><price.amount>-5.0</price.amount></Point>
    <Point><position>2</position><price.amount>42.5</price.amount></Point>
    </Period></TimeSeries></Publication_MarketDocument>"""
    result = parse_timeseries(xml, "price.amount")
    assert result.value.tolist() == [-5.0, 42.5]
    assert result.timestamp_utc.tolist() == [
        pd.Timestamp("2024-01-01T00:00Z"),
        pd.Timestamp("2024-01-01T01:00Z"),
    ]


def test_parse_filters_classification_position() -> None:
    xml = b"""<Publication_MarketDocument xmlns="urn:test">
    <TimeSeries><classificationSequence_AttributeInstanceComponent.position>1</classificationSequence_AttributeInstanceComponent.position><Period>
    <timeInterval><start>2024-01-01T00:00Z</start></timeInterval><resolution>PT60M</resolution>
    <Point><position>1</position><price.amount>50</price.amount></Point></Period></TimeSeries>
    <TimeSeries><classificationSequence_AttributeInstanceComponent.position>2</classificationSequence_AttributeInstanceComponent.position><Period>
    <timeInterval><start>2024-01-01T00:00Z</start></timeInterval><resolution>PT15M</resolution>
    <Point><position>1</position><price.amount>999</price.amount></Point></Period></TimeSeries>
    </Publication_MarketDocument>"""
    result = parse_timeseries(xml, "price.amount", classification_position="1")
    assert result.value.tolist() == [50.0]
