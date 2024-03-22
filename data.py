import numpy as np
import pandas as pd
import json
from scipy.sparse import coo_matrix


class Dataset:

    def __init__(self, name):

        self.name = name
        self.disease = name.split('.')[0]

        # genotype

        npz = np.load('method1/genotype.%s.npz' % name, allow_pickle=True)

        self.genotype = pd.DataFrame(npz['genotype'], index=npz['SNPs'], columns=npz['eids'])

        # GWAS

        self.GWAS = pd.read_csv('method1/GWAS.%s.csv' % name, dtype={'Identifier': str})

        self.GWAS.set_index('Identifier', inplace=True)

        # phenotype

        js = json.load(open('method1/filtered_disease_eids_noproxy.json'))
        # js = json.load(open('method1/filtered_disease_eids_nokinship_noproxy.json'))

        cases, controls = pd.Series(1, index=js[self.disease]['case']), pd.Series(0, index=js[self.disease]['control'])
        self.phenotype = pd.concat([cases, controls])

        # LD
        self.r2 = None

        # genetic PCs
        self.PCs = None

        # covariates
        self.covariates = None

        # gene association
        self.gene_association = None

        # gene interaction
        self.gene_interaction = None

    # lazy loader
    def load(self, name):

        if name == 'r2' and self.r2 is None:
            js = json.load(open('method1/LD.%s.json' % self.name))
            self.r2 = {float(r2): set(snps.keys()) for r2, snps in js.items()}
            self.r2[-1] = set(self.genotype.index)
        elif name == 'PCs' and self.PCs is None:
            self.PCs = pd.read_csv('method1/22009.csv', dtype={'eid': str})
            self.PCs.set_index('eid', inplace=True)
            self.PCs.dropna(inplace=True)
        elif name == 'covariates' and self.covariates is None:
            self.covariates = pd.read_csv('method1/covariates.csv', dtype={'eid': str})
            self.covariates.set_index('eid', inplace=True)
        elif name == 'gene_association' and self.gene_association is None:
            self.gene_association = pd.read_csv('method1/GA.UniProt.csv')
            # self.gene_association = pd.read_csv('GA.UniProt.GO.csv')
            # self.gene_association = pd.read_csv('GA.Ensembl.eQTL.csv')
            # self.gene_association = pd.read_csv('GA.Ensembl.PPI.csv')
            snps = set(self.genotype.index)
            self.gene_association.query('Identifier in @snps', inplace=True)
        elif name == 'gene_interaction' and self.gene_interaction is None:
            npz = np.load('method1/PPI.String.npz', allow_pickle=True)
            self.interaction_genes = npz['proteins']
            self.gene_interaction = (npz['experimental'], (npz['rows'], npz['cols']))
            # self.gene_interaction = (npz['combined_score'], (npz['rows'], npz['cols']))
            self.gene_interaction = coo_matrix(self.gene_interaction, shape=(
            self.interaction_genes.shape[0], self.interaction_genes.shape[0])).toarray()
            self.gene_interaction += self.gene_interaction.T

    def filter(self, p_value=5e-4, r2=-1):

        # GWAS = self.GWAS.query('Pvalue < %e & Pvalue > 5e-8' % p_value)

        GWAS = self.GWAS.query('Pvalue < %e ' % p_value)


        snps = GWAS.index.intersection(self.genotype.index)

        if r2 > 0:
            self.load('r2')
            snps = snps.intersection(self.r2[r2])

        return snps.drop_duplicates()

    def association(self, SNPs, return_matrix=True):

        self.load('gene_association')

        gene_association = self.gene_association.query('Identifier in @ SNPs')

        # a = {gene_id: set(g['Identifier']) for gene_id, g in gene_association.groupby('EnsemblProteinID', sort=False)}
        # a = {gene_id: set(g['Identifier']) for gene_id, g in gene_association.groupby('EnsemblID', sort=False)}
        a = {gene_id: set(g['Identifier']) for gene_id, g in gene_association.groupby('UniProtID', sort=False)}

        a = [({i}, j) for i, j in a.items()]

        while True:

            a = sorted(a, key=lambda x: frozenset(x[0]), reverse=True)
            a = sorted(a, key=lambda x: len(x[1]), reverse=True)

            mp = dict()
            for i in range(1, len(a)):
                ai = a[i]
                for j in range(i):
                    aj = a[j]
                    if len(ai[1] & aj[1]) > (float(len(ai[1])) * 0.95):
                        mp[i] = j
                        break

            if len(mp) == 0:
                break

            for i, j in mp.items():
                a[j][0].update(a[i][0])
                a[j][1].update(a[i][1])

            a = [j for i, j in enumerate(a) if i not in mp]

        groups = a

        if not return_matrix:
            mp = {j: i for i, j in enumerate(SNPs)}
            return list(map(lambda x: list(map(mp.get, x[1])), groups)), [i[0] for i in groups]

        mp = {i: set() for i in SNPs}
        for i, g in enumerate(groups):
            for snp in g[1]:
                mp[snp].add(i)

        mask = np.zeros((len(SNPs), len(groups)))

        for i, snp in enumerate(SNPs):
            for j in mp[snp]:
                mask[i, j] = 1

        return mask, [i[0] for i in groups]

    def interaction(self, gene_groups):

        self.load('gene_interaction')

        mp = {j: i for i, j in enumerate(self.interaction_genes)}

        gene_groups = list(map(lambda x: list(map(mp.get, x)), gene_groups))

        mask = np.zeros((len(gene_groups), len(gene_groups)))
        for i, group_i in enumerate(gene_groups):
            interaction_i = self.gene_interaction[group_i]
            for j, group_j in enumerate(gene_groups):
                # mask[i, j] = interaction_i[:, group_j].max()
                mask[i, j] = interaction_i[:, group_j].sum() / float(len(group_i) * len(group_j))

        return mask

    def extract(self, SNPs=None, eids=None, return_PCs=False, return_covariates=False):

        # SNPs

        SNPs = self.genotype.index if (SNPs is None) else self.genotype.index.intersection(SNPs)

        # samples

        eids = (self.phenotype.index if (eids is None) else eids).intersection(self.genotype.columns)

        # genotype

        genotype = self.genotype.loc[SNPs, eids]

        # phenotype

        phenotype = self.phenotype[eids]

        # GWAS

        beta = self.GWAS['Beta'].loc[SNPs]

        result = [genotype, phenotype, beta]

        # genetic PCs
        if return_PCs:
            self.load('PCs')
            PCs = self.PCs.loc[eids]
            result.append(PCs)

        # covariates
        if return_covariates:
            self.load('covariates')
            covariates = self.covariates.loc[eids, ['sex', 'platform', 'age', 'degree']]
            result.append(covariates)

        return result


if __name__ == "__main__":
    import sys

# dataset = Dataset(sys.argv[1])

# eQTL = pd.read_csv('GTEx.eQTL.csv', index_col=['Identifier'])
# eQTL.query('PVALUE_FE < 5e-2', inplace=True)

# result = list()

# for p_value in [5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8]:

# 	nums = [p_value]
# 	for r2 in [-1, 0.8, 0.6, 0.4, 0.2]:
# 		snps = dataset.filter(p_value, r2)
# 		snps = snps.intersection(eQTL.index).drop_duplicates()
# 		nums.append(len(snps))
# 		# nums.append(dataset.association(snps).shape)

# 	result.append(nums)

# result = pd.DataFrame(result, columns=['p_value', -1, 0.8, 0.6, 0.4, 0.2])
# print(result)

