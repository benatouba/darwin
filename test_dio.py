from constants import basepath
from dio import extract_xml_metadata


def test_extract_xml_metadata():
    metadata = extract_xml_metadata(
        f"{basepath}darwin_measured/11-AWS-PCerroCrocker_complete/11_AWS-PCerroCrocker_complete_meta.xml"
    )
    assert len(metadata.rows) > 0
