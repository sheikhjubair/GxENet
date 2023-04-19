#!/usr/bin/env python
# coding: utf-8

# In[ ]:


version ='v4'
training_data = '../processed_data/tr_geno_with_8_m_avg_all_data_' + version + '.pkl'
test_data = '../processed_data/test_geno_with_8_m_avg_all_data_{}.pkl'.format(version)
test_unique_env_data = '../processed_data/test_geno_unique_env_with_8_m_avg_all_data_{}.pkl'.format(version)
validation_data = '../processed_data/val_geno_with_8_m_avg_all_data_{}.pkl'.format(version)
soil_data = '../processed_data/soil_info.pkl'.format(version)

model_path = '../outputs/fc_model_env_middle_version_{}.pt'.format(version)
model_avg_by_geno_path = '../outputs/fc_model_avg_by_geno_version_{}.pt'.format(version)
model_avg_by_env_path = '../outputs/fc_model_avg_by_env_version_{}.pt'.format(version)
final_model_path = '../outputs/fc_model_final_version_{}.pt'.format(version)
soil_model_path = '../outputs/fc_model_final_version_soil_{}.pt'.format(version)

target_scaler_path = '../outputs/target_scaler_version_{}.pt'.format(version)
weather_scaler_path = '../outputs/weather_scaler_version_{}.pt'.format(version)
fc_training_loss = '../outputs/fc_training_loss_version_{}'.format(version)
fc_validation_loss = '../outputs/fc_validation_loss_version_{}'.format(version)
target_scaler = '../outputs/target_scaler_{}.pkl'.format(version)
weather_scaler = '../outputs/weather_scaler_.pkl'.format(version)

training_representation = '../processed_data/tr_rep_learned_' + version + '.pkl'
test_representation = '../processed_data/test_rep_learned_' + version + '.pkl'
val_representation = '../processed_data/val_rep_learned_' + version + '.pkl'
test_representation_unique_env = '../processed_data/test_unique_env_rep_learned_' + version + '.pkl'

# In[ ]:


num_env = 81
num_epoch = 3000
batch_size = 32
fc_model_reduce_dim = 750

