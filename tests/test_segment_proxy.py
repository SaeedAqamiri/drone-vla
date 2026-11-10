import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_segment_proxy import summarize


class SegmentProxyTest(unittest.TestCase):
    def test_pass_requires_every_frame_accepted_and_accurate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frames.csv'
            rows=[{'timestamp':str(i),'accepted':'True','error_m':'12'} for i in range(3)]
            def write():
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            write()
            self.assertTrue(summarize(path,30,3)['all_frames_within_threshold'])
            rows[1]['accepted']='False'; rows[1]['error_m']=''
            write()
            result=summarize(path,30,3)
            self.assertFalse(result['all_frames_within_threshold'])
            self.assertEqual(result['no_accepted_error'],1)
            rows[1]['timestamp']='0'; write()
            with self.assertRaisesRegex(ValueError,'timestamp'):
                summarize(path,30,3)

if __name__=='__main__': unittest.main()
