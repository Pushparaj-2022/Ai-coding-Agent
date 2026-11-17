from lca.providers.toolcall_parser import StreamFilter, extract_tool_calls, parse_pythonic_calls


def test_json_tag():
    calls, cleaned = extract_tool_calls('Looking.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>')
    assert cleaned == "Looking."
    assert [(c.name, c.arguments, c.error) for c in calls] == [("read_file", {"path": "a.py"}, None)]


def test_multiple_calls_and_alternative_keys():
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"tool":"b","parameters":{"x":1}}</tool_call>'
    calls, _ = extract_tool_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [("a", {}), ("b", {"x": 1})]


def test_arguments_as_json_string():
    calls, _ = extract_tool_calls('<tool_call>{"name":"a","arguments":"{\\"p\\": 2}"}</tool_call>')
    assert calls[0].arguments == {"p": 2}


def test_unterminated_and_trailing_commas():
    calls, cleaned = extract_tool_calls('ok <tool_call>{"name": "list_files", "arguments": {"glob": "*.py",},}')
    assert cleaned == "ok"
    assert calls[0].name == "list_files" and calls[0].arguments == {"glob": "*.py"}


def test_lfm_pythonic_format():
    text = """<|tool_call_start|>[read_file(path="src/a.py", start_line=10), search_code(pattern='def \\\\w+', case_sensitive=True)]<|tool_call_end|>"""
    calls, cleaned = extract_tool_calls(text)
    assert cleaned == ""
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "src/a.py", "start_line": 10}
    assert calls[1].arguments == {"pattern": "def \\w+", "case_sensitive": True}


def test_alias_tags_used_by_fine_tuned_qwen():
    calls, cleaned = extract_tool_calls('Checking.\n<tools>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tools>')
    assert cleaned == "Checking."
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.py"})]
    calls, _ = extract_tool_calls('<json>{"name": "list_files", "arguments": {}}</json>')
    assert calls[0].name == "list_files"


def test_alias_tags_without_a_call_stay_text():
    text = "Wrap it like <json>{\"a\": 1}</json> or <tools>none</tools>."
    assert extract_tool_calls(text) == ([], text)


def test_pythonic_literals_lists_dicts():
    [c] = parse_pythonic_calls('f(a=[1, 2.5], b={"k": None}, c=False, d="""multi\nline""")')
    assert c.arguments == {"a": [1, 2.5], "b": {"k": None}, "c": False, "d": "multi\nline"}


def test_garbage_becomes_invalid_call_not_exception():
    calls, _ = extract_tool_calls("<tool_call>this is not a call</tool_call>")
    assert calls[0].name == "__invalid__" and calls[0].error


def test_non_object_arguments_reported():
    calls, _ = extract_tool_calls('<tool_call>{"name":"a","arguments":[1,2]}</tool_call>')
    assert calls[0].error == "arguments must be a JSON object"


def test_bare_json_only_for_known_tools():
    text = '{"name": "read_file", "arguments": {"path": "x"}}'
    assert extract_tool_calls(text)[0] == []
    calls, cleaned = extract_tool_calls(text, {"read_file"})
    assert calls[0].arguments == {"path": "x"} and cleaned == ""
    assert extract_tool_calls('{"name": "evil", "arguments": {}}', {"read_file"})[0] == []


def test_plain_text_untouched():
    calls, cleaned = extract_tool_calls("Use `a < b` and <div> tags.")
    assert calls == [] and cleaned == "Use `a < b` and <div> tags."


def test_stream_filter_hides_split_tags():
    out = []
    f = StreamFilter(out.append)
    for chunk in ["Hi <to", 'ol_call>{"x":1}</tool', "_call> there <th", "ink>secret</think>!"]:
        f.push(chunk)
    f.end()
    assert "".join(out) == "Hi  there !"


def test_stream_filter_keeps_ordinary_angle_brackets():
    out = []
    f = StreamFilter(out.append)
    f.push("if a < b: return <div>")
    f.end()
    assert "".join(out) == "if a < b: return <div>"


def test_gemma_native_call_with_json_args():
    calls, cleaned = extract_tool_calls('Checking.\n<|tool_call>call:get_time{"timezone":"UTC"}<tool_call|>')
    assert [(c.name, c.arguments, c.error) for c in calls] == [("get_time", {"timezone": "UTC"}, None)]
    assert cleaned == "Checking."


def test_gemma_native_call_with_gemma_quoting_and_no_closing_tag():
    calls, cleaned = extract_tool_calls('<|tool_call>call:read_file{path:<|"|>src/a "b".py<|"|>,limit:5,deep:true}')
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": 'src/a "b".py', "limit": 5, "deep": True}
    assert cleaned == ""


def test_gemma_multiple_calls_and_nested_args():
    text = '<|tool_call>call:a{}<tool_call|><|tool_call>b{"x":{"y":[1,"}"]}}<tool_call|> done'
    calls, cleaned = extract_tool_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [("a", {}), ("b", {"x": {"y": [1, "}"]}})]
    assert cleaned == "done"


def test_gemma_unterminated_args_is_reported():
    calls, _ = extract_tool_calls('<|tool_call>call:a{"x": 1')
    assert calls[0].name == "a" and calls[0].error


def test_stream_filter_hides_gemma_markup():
    from lca.providers.toolcall_parser import StreamFilter

    out: list[str] = []
    f = StreamFilter(out.append)
    for chunk in ["Hi <|tool", '_call>call:a{"x":1}<tool_', "call|> bye"]:
        f.push(chunk)
    f.end()
    assert "".join(out) == "Hi  bye"
