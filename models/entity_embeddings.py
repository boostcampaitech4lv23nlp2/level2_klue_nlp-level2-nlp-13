import torch
from torch import nn


class CustomRobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            32024, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        ###### Added Entity Embedding Layer ######
        ## 0:Niether, 1:subject, 2:object, 3~10: <S:{type}> <O:{type}> type:["PER", "ORG", "POH", "DAT", "LOC", "NOH"]
        self.entity_type_embeddings = nn.Embedding(11, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        ##### Entity type 정보를 주는 Embedding layer 추가  ######
        # if input_ids is not None:

        entity_ids = self.create_entity_ids_from_input_ids(input_ids)
        entity_type_embeddings = self.entity_type_embeddings(entity_ids)
        embeddings += entity_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def create_position_ids_from_input_ids(
        self, input_ids, padding_idx, past_key_values_length=0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.
        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    def create_entity_ids_from_input_ids(self, input_ids):
        # <S:ORG> 32004 </S:ORG> 32005
        # <S:PER> 32000 </S:PER> 32001
        # <O:ORG> 32006 </O:ORG> 32007
        # <O:PER> 32002 </O:PER> 32003
        # <O:POH> 32010 </O:POH> 32011
        # <O:DAT> 32014 </O:DAT> 32015
        # <O:LOC> 32018 </O:LOC> 32019
        # <O:NOH> 32022 </O:NOH> 32023

        # fmt: off
        mapping_dict = {32004: 3, 32005: 3, 32000: 4, 32001: 4, 32006: 5, 32007: 5, 32002: 6, 32003: 6, 32010: 7, 32011: 7, 32014: 8, 32015: 8, 32018: 9, 32019: 9, 32022: 10, 32023: 10}
        # fmt: off

        s_entity_input_ids = [[32004, 32005], [32000, 32001]]
        o_entity_input_ids = [
            [32006, 32007],
            [32002, 32003],
            [32010, 32011],
            [32014, 32015],
            [32018, 32019],
            [32022, 32023],
        ]
        s_ids, o_ids = [], []

        for i, j in s_entity_input_ids:
            s_1 = torch.nonzero((input_ids == i))
            s_2 = torch.nonzero((input_ids == j))
            temp = torch.cat([s_1, s_2], dim=1).tolist()
            s_ids.extend(temp)

        for i, j in o_entity_input_ids:
            o_1 = torch.nonzero((input_ids == i))
            o_2 = torch.nonzero((input_ids == j))
            temp = torch.cat([o_1, o_2], dim=1).tolist()
            o_ids.extend(temp)

        entity_ids = torch.zeros_like(input_ids)

        for i in range(len(s_ids)):
            s_id = s_ids[i]
            entity_ids[s_id[0], s_id[1]] = mapping_dict[input_ids[s_id[0], s_id[1]].item()]
            entity_ids[s_id[2], s_id[3]] = mapping_dict[input_ids[s_id[2], s_id[3]].item()]
            entity_ids[s_id[0], s_id[1] + 1 : s_id[3]] = 1

        for i in range(len(o_ids)):
            o_id = o_ids[i]
            entity_ids[o_id[0], o_id[1]] = mapping_dict[input_ids[o_id[0], o_id[1]].item()]
            entity_ids[o_id[2], o_id[3]] = mapping_dict[input_ids[o_id[2], o_id[3]].item()]
            entity_ids[o_id[0], o_id[1] + 1 : o_id[3]] = 2

        return entity_ids
