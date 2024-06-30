ner_prompt = """**Objective:** Produce realistic text passages that include clearly identified named entities. Each 
    entity should be meticulously labeled according to its type for straightforward extraction. 

**Format Requirements:**
- The output should be formatted in JSON, containing the text and the corresponding entities list.
- Each entity in the text should be accurately marked and annotated in the 'entities' list.
- Meticulously follow all the listed attributes.

**Entity Annotation Details:**
- All entity types must be in lowercase. For example, use "type" not "TYPE".
- Entity types can be multiwords separate by space. For instance, use "entity type" rather than "entity_type".
- Entities spans can be nested within other entities.
- A single entity may be associated with multiple types. list them in the key "types".

**Output Schema:**

<start attribute_1="value1" attribute_2="value2" ...>
{
  "text": "{text content}",
  "entities": [
    {"entity": "entity name", "types": ["type 1", "type 2", ...]},
    ...
  ]
}
<end>

**Give me the real world example that is following this instructions, don't give any explains, just return output like Output Schema**:"""