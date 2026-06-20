import torch
from alignment_utils import parse_alignment_from_sequences

class TMLDDTLoss(torch.nn.Module):
    """
    Custom loss function that combines TM and LDDT scores based on TM score ranges.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for per-residue loss
        self.beta = beta    # Weight for global loss
        
    def forward(self, new_emb1, new_emb2, global_emb1, global_emb2, 
                tm_scores, lddt_scores, seqxA_list, seqM_list, seqyA_list,
                min_tm_score_for_global=0.1, min_tm_score_for_lddt=0.6):
        """
        Compute loss based on TM score ranges.
        
        Args:
            new_emb1, new_emb2: Per-residue embeddings
            global_emb1, global_emb2: Global embeddings
            tm_scores: TM scores for each protein pair
            lddt_scores: LDDT scores for each residue
            seqxA_list: List of seqxA strings for each pair
            seqM_list: List of seqM strings for each pair
            seqyA_list: List of seqyA strings for each pair
            min_tm_score_for_global: Minimum TM-score for global loss
            min_tm_score_for_lddt: Minimum TM-score for lDDT loss
        """
        batch_size = tm_scores.shape[0]
        total_loss = 0.0
        total_residue_loss = 0.0
        total_global_loss = 0.0
        valid_pairs = 0
        valid_lddt_pairs = 0
        
        for i in range(batch_size):
            tm_score = tm_scores[i].item()
            
            # # Skip pairs with TM score < min_tm_score_for_global
            # if tm_score < min_tm_score_for_global:
            #     continue
            
            valid_pairs += 1
            
            # Compute cosine similarity for global embeddings
            global_sim = torch.cosine_similarity(global_emb1[i], global_emb2[i], dim=0)
            
            # Target similarity based on TM score
            target_sim = torch.tensor(tm_score, device=global_sim.device)
            
            # Global loss (always use TM score)
            global_loss = torch.nn.functional.l1_loss(global_sim, target_sim)
            
            # Per-residue loss
            if tm_score >= min_tm_score_for_lddt:
                # Use combination of TM and LDDT scores with alignment
                residue_loss = self._compute_residue_loss_with_lddt(
                    new_emb1[i], new_emb2[i], lddt_scores[i], 
                    seqxA_list[i], seqM_list[i], seqyA_list[i]
                )
                pair_loss = self.alpha * residue_loss + self.beta * global_loss
                total_residue_loss += residue_loss
                valid_lddt_pairs += 1
            else:
                # Only global loss
                pair_loss = global_loss
            
            # Apply TM-score weighting
            weighted_global_loss = global_loss
            weighted_pair_loss = pair_loss
            
            total_loss += weighted_pair_loss
            total_global_loss += weighted_global_loss

        avg_loss = total_loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=new_emb1.device)
        avg_global_loss = total_global_loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=new_emb1.device)
        
        if valid_lddt_pairs == 0:
            return avg_global_loss, {
                'residue_loss': torch.tensor(0.0, device=new_emb1.device),
                'global_loss': avg_global_loss,
            }
        
        avg_residue_loss = total_residue_loss / valid_lddt_pairs
        
        return avg_loss, {
            'residue_loss': avg_residue_loss,
            'global_loss': avg_global_loss,
        }
    
    def _compute_residue_loss_with_lddt(self, emb1, emb2, lddt_scores, seqxA, seqM, seqyA):
        """Compute per-residue loss using LDDT scores and alignment from sequences."""
        # Parse alignment from sequences
        alignment = parse_alignment_from_sequences(seqxA, seqM, seqyA)
        
        if not alignment:
            return torch.tensor(0.0, device=emb1.device)
        
        # parse_alignment_from_sequences(seqxA, seqM, seqyA) yields (ref_idx, model_idx)
        # with seqxA = reference (TM-align chain 1), seqyA = model (chain 2).
        ref_indices, model_indices = zip(*alignment)
        L1, L2 = emb1.shape[0], emb2.shape[0]
        Lt = lddt_scores.shape[0]
        pairs = [
            (int(r), int(m))
            for r, m in zip(ref_indices, model_indices)
            if 0 <= int(r) < L1 and 0 <= int(m) < L2 and 0 <= int(m) < Lt
        ]
        if not pairs:
            return torch.tensor(0.0, device=emb1.device)

        ref_ix = torch.tensor([p[0] for p in pairs], device=emb1.device, dtype=torch.long)
        mod_ix = torch.tensor([p[1] for p in pairs], device=emb1.device, dtype=torch.long)

        # emb1 / seq1 = reference, emb2 / seq2 = model; lddt padded vector is indexed by model residue
        aligned_emb1 = emb1[ref_ix]
        aligned_emb2 = emb2[mod_ix]
        aligned_targets = lddt_scores[mod_ix].to(emb1.device)
        
        # Compute cosine similarities between aligned residues
        similarities = torch.cosine_similarity(aligned_emb1, aligned_emb2, dim=1)
        
        residue_loss = torch.nn.functional.l1_loss(similarities, aligned_targets)
        
        return residue_loss