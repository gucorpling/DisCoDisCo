from gucorpling_models.dataset_reader import Disrpt2021Reader


class TestTextClassificationJsonReader:
    def test_read_from_eng_rst_gum_sample(self):
        reader = Disrpt2021Reader()
        data_path = "tests/fixtures/toy_data.rels"
        instances = list(reader.read(data_path))

        assert len(instances) == 2
        print(instances[0])
        print(instances[1])

        fields = instances[0].fields
        expected_tokens = ["In", "the", "present", "study", ","]
        assert [t.text for t in fields["unit2_body"].tokens][:5] == expected_tokens
        assert fields["relation"].label == "preparation"

        fields = instances[1].fields
        expected_tokens = ["Research", "on", "adult-learned", "second", "language"]
        print([t.text for t in fields["unit1_body"].tokens])
        assert [t.text for t in fields["unit1_body"].tokens][:5] == expected_tokens
        assert fields["relation"].label == "background"
