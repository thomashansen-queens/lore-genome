from fastapi.datastructures import FormData
from pydantic import BaseModel, Field

from lore.web.utils.forms import form_html_to_dict


class DefaultsModel(BaseModel):
    name: str = "default name"
    count: int = 3
    tags: list[str] = Field(default_factory=lambda: ["default-tag"])


def test_form_html_to_dict_uses_defaults_for_blank_values():
    form = FormData([
        ("name", ""),
        ("count", "   "),
        ("tags", ""),
    ])

    result = form_html_to_dict(form, DefaultsModel, blank_to_default=True)

    assert result == {
        "name": "default name",
        "count": 3,
        "tags": ["default-tag"],
    }
