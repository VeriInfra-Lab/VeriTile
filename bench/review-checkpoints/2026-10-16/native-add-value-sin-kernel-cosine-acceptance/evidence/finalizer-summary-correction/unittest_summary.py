"""Check actual unittest completion without treating buffered stdout as status."""
import re


def observed_unittest_count(text, exit_code):
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError('Unittest command did not exit successfully')
    runs = re.findall(r'^Ran (\d+) tests? in [^\r\n]+$', text, re.M)
    status = re.findall(r'^(?:OK(?: \([^\r\n]*\))?|FAILED(?: \([^\r\n]*\))?)$', text, re.M)
    completed = re.findall(r'^Ran (\d+) tests? in [^\r\n]+\n\nOK\s*$', text, re.M)
    if (len(runs) != 1 or int(runs[0]) <= 0 or status != ['OK']
            or completed != runs):
        raise ValueError('Unclear unittest completion')
    return int(runs[0])
