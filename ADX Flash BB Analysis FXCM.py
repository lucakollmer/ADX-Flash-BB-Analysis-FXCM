# %% Preamble.

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:23:01 2022

@author: lucak
"""

# %% Import Packages.

import numpy as np
import pandas as pd
from tqdm import tqdm

# pip install ta
import ta.trend
import ta.volatility

# %% Import Historical Data from FXCM HDD Basic.

csv1 = pd.read_csv(r'E:\Google Drive\Python\Quantitative Finance\ADX Flash BB Analysis FXCM\EURUSD_m1_Y1_BidAndAsk.csv')
                   
# %% Transform FXCM HDD Basic Price Series to OHLC.

def transform_data(df):
    
    df = df.copy()
    
    # Combine Date and Time, then drop Date.
    df['Time'] = df['Date'] + " " + df['Time']
    df = df.iloc[:,1:]
    
    # Rename Bid OHLC to OHLC
    df['Open'] = df['OpenBid']
    df['High'] = df['HighBid']
    df['Low'] = df['LowBid']
    df['Close'] = df['CloseBid']
    
    # Drop Bid and Ask OHLC and rename Total Ticks to Volume.
    df = df.drop(columns = ['OpenBid', 
                            'HighBid', 
                            'LowBid', 
                            'CloseBid',
                            'OpenAsk', 
                            'HighAsk', 
                            'LowAsk', 
                            'CloseAsk'
                            ])
    
    df = df.rename(columns = {'Total Ticks': 'Vol'})    
    
    return df

# %% Generate ADX Flash BB Strategy Trading Signals.

def adx_flash_bb(df,
                 adx_lookback,
                 adx_threshold,
                 bb_lookback,
                 bb_closes
                 ):
    
    df = df.copy()
    
    # Generate ADX Flash signals using ta library.
    df['ADX'] = ta.trend.adx(df['High'], 
                             df['Low'], 
                             df['Close'], 
                             adx_lookback,
                             False
                             )
    
    df['Flash'] = np.where(((df['ADX'] < adx_threshold) 
                            & (df['ADX'] > 0)), 1, 0)
    
    df['Signal'] = df['Flash'].diff()
    
    # Generate Bollinger Bands Signals using ta library.
    df['BBHigh'] = ta.volatility.bollinger_hband(df['Close'], 
                                                 bb_lookback,
                                                 2, 
                                                 False
                                                 )
    
    df['BBLow'] = ta.volatility.bollinger_lband(df['Close'], 
                                                bb_lookback, 
                                                2, 
                                                False
                                                )
    df['BBSignal'] = 0
    df['BBSignal'] = np.where(((df['Close'] < df['BBHigh']) 
                               & (df['Close'] > df['BBLow'])), 0, 1)  
    
    # Iterate through price series to generate BB signal.
    x = 0
    for i, row in tqdm(df.iterrows()):
        x = np.where(row.BBSignal == 1, x + 1, 0)
        df.at[i, 'BBSignal'] = x
        
    # ADX values only accurate after 200 rows.
    df = df.iloc[200:, :]
    
    return df

# %% Function to convert lists of ADX Flashes to pandas DataFrames.

def convert_list_to_dataframe(li):
    
    df = pd.DataFrame(li, columns=['Time', 
                                   'Age', 
                                   'Stage', 
                                   'FO', 
                                   'FH', 
                                   'FL', 
                                   'FD',
                                   'WO', 
                                   'WH', 
                                   'WL', 
                                   'WD', 
                                   'HO', 
                                   'HH', 
                                   'HL', 
                                   'HD', 
                                   'Bias', 
                                   'MaxP', 
                                   'MaxL'
                                   ]
                      )
    
    return df

# %% Main engine to analyse the ADX Flash BB strategy.

def adx_flash_analysis(df,
                       adx_lookback,
                       adx_threshold,
                       grace_period,
                       bb_lookback,
                       bb_closes
                       ):
    
    df = df.copy()
    
    # Transform price series and generate ADX Flash BB signals.
    df = adx_flash_bb(transform_data(df), 
                      adx_lookback, 
                      adx_threshold,
                      bb_lookback,
                      bb_closes)

    # Initialise empty lists.
    open_flashes = []
    closed_flashes = []
    closed_flashes_queue = []
    stunted_flashes = []
    stunted_flashes_queue = []
    current_flash = []    
    
    # Initialise measurement integers.
    bulls = 0
    bears = 0
    bullsT = 0
    bearsT = 0
    stunted = 0
    
    # Initialise measurement columns in price series (df).
    df['Bulls'] = 0
    df['Bears'] = 0
    df['BullsT'] = 0
    df['BearsT'] = 0
    df['Stunted'] = 0   
    
    # Iterate through price series (df).
    for row in tqdm(df.itertuples()):
        
        # Iterate through the list open_flashes.
        for flash in open_flashes:
            
            # Update flash age by +1.
            flash[1] += 1
            
            # Flash State Progression:
            # 0 = Flash,
            # 1 = Window,
            # 2 = Hold,
            # 3 = Closed.
            
            # Check Stage 2 (Hold) Flashes.
            if (flash[2] == 2):
                
                # Increase Hold Duration (HD) by 1.
                flash[14] += 1 
                
                # Progress Flash from Stage 2 to Stage 3 if price returns 
                # to Window Open (WO).
                if (flash[10] >= grace_period 
                    and not (row.High < flash[7] 
                             or row.Low > flash[7])):
                    
                    # Progress Stage and move to closed_flashes list.
                    flash[2] += 1 
                    closed_flashes_queue.append(open_flashes.index(flash))
                    
                    # Check Bias of Flash and increase corresponding measure.
                    if (flash[15] == 1):
                        
                        bulls -= 1 # Decrease Active Bear Flashes by 1.
                        bullsT += 1 # Increase Closed Bear Flashes by 1.
                    
                    else:
                        
                        bears -= 1 # Decrease Active Bull Flashes by 1.
                        bearsT += 1 # Increase Closed Bull Flashes by 1.
                
                # Update Max Return and Max Drawdown.
                else:
                    
                    flash[12] = np.where(row.High > flash[12], 
                                         row.High, 
                                         flash[12]
                                         )
                    
                    flash[13] = np.where(row.Low < flash[13], 
                                         row.Low,
                                         flash[13]
                                         )
                    
            # Check Stage 1 (Window) Flashes.
            elif (flash[2] == 1):
                
                # Set flash bias after grace_period.
                if (flash[10] == grace_period):
                    
                    if (row.Open <= flash[7]):
                        
                        flash[15] = 1 # Set bias Bullish.
                        bulls += 1 # Increment Active Bull Flashes by 1.
                    
                    else:
                        
                        flash[15] = -1 # Set bias Bearish.
                        bears += 1 # Increment Active Bear Flashes by 1.
                
                # Progress from Stage 1 to Stage 3 if price returns to 
                # Window Open (WO) or...
                if (flash[10] >= grace_period 
                    and not (row.High < flash[7] 
                             or row.Low > flash[7])):
                    
                    flash[10] += 1 # Increase Window Duration (WD) by 1.
                    flash[2] += 2 # Progress Stage.
                    
                    for i in range(11,14): # Set Hold OHL to WO.
                    
                        flash[i] = flash[7]
                    
                    closed_flashes_queue.append(open_flashes.index(flash))
                    
                    if (flash[15] == 1):
                        
                        bulls -= 1 # Decrease Active Bear Flashes by 1.
                        bullsT += 1 #I ncrease Closed Bear Flashes by 1.
                   
                    else:
                        
                        bears -= 1 # Decrease Active Bull Flashes by 1.
                        bearsT += 1 # Increase Closed Bull Flashes by 1.
                    
                
                #...progress from Stage 1 to Stage 2 if another Flash appears.
                elif (row.Signal == 1):
                    
                    #Check if Window was open longer than grace_period.
                    if (flash[10] < grace_period):
                        
                        stunted_flashes_queue.append(open_flashes.index(flash))
                        stunted += 1 # Increase Total Stunted Flashes by 1.
                    
                    else:
                        
                        flash[2] += 1 # Progress Stage.
                        
                        # Set inital Hold OHL and Duration.
                        flash[11] = row.Open # Set Hold Open (HO).
                        flash[12] = row.High
                        flash[13] = row.Low
                        flash[14] = 1
                
                else:
                    
                    # Update Window High, Low and Duration (WH, WL, WD).
                    flash[8] = np.where(row.High > flash[8], 
                                        row.High, 
                                        flash[8]
                                        )
                    
                    flash[9] = np.where(row.Low < flash[9], 
                                        row.Low, 
                                        flash[9]
                                        )
                    
                    flash[10] += 1 # Increase Window Duration (WD) by 1.            

            # Check Stage 0 (Flash) Flashes.
            elif (flash[2] == 0):
                
                # Progress from Stage 0 to Stage  1 once flash has ended.
                if (row.Signal == -1):
                    
                    flash[2] += 1 # Progress Stage.
                    
                    #Set initial Window OHL and Duration.
                    
                    # Set Window Open (WO): Flash Target Price.
                    flash[7] = row.Open 
                    
                    flash[8] = row.High
                    flash[9] = row.Low
                    flash[10] = 1         
                    
                else:
                    # Update Flash High, Low and Duration (FH, FL, FD).
                    flash[4] = np.where(row.High > flash[4], 
                                        row.High, 
                                        flash[4]
                                        )
                    
                    flash[5] = np.where(row.Low < flash[5],
                                        row.Low,
                                        flash[5]
                                        )
                    
                    flash[6] += 1 # Increase Flash Duration (FD) by 1.


        # Pop flashes from lists closed/stunted_flashes_queue to 
        # lists closed/stunded_flashes.
        for i in reversed(closed_flashes_queue):
            
            closed_flashes.append(open_flashes.pop(i))
            
        closed_flashes_queue = []
        
        for i in reversed(stunted_flashes_queue):
            
            stunted_flashes.append(open_flashes.pop(i))
            
        stunted_flashes_queue = []
            
        # Update series measures.
        df.at[row.Index, 'Bulls'] = bulls
        df.at[row.Index, 'Bears'] = bears
        df.at[row.Index, 'BullsT'] = bullsT
        df.at[row.Index, 'BearsT'] = bearsT
        df.at[row.Index, 'Stunted'] = stunted
        
        # Create New Flash.
        if (row.Signal == 1):
            
            current_flash = [row.Time, 1] # Set flash initiation time and age.
            
            for i in range(16): # Generate empty columns
            
                current_flash.append(0)
                
            #Set inital Flash OHL.
            current_flash[3] = row.Open
            current_flash[4] = row.High
            current_flash[5] = row.Low
            current_flash[6] = 1
            open_flashes.append(current_flash)
            current_flash = []
            
    # Calculate Max Profit and Drawdowns of Flashes.
    for flash in closed_flashes:
        
        if (flash[15] > 0): # Bullish
            
            # Maximum Profit is WO - WL.
            flash[16] = (flash[7] - flash[9])/flash[9] 
            
            #Maximum Loss is WO - min(WL, HL).
            flash[17] = (flash[7] - min(flash[9],
                                        flash[13])
                         )/min(flash[9],
                               flash[13]
                               )
            
        else: # Bearish
        
            # Maximum Profit is WH - WO.
            flash[16] = (flash[8] - flash[7])/flash[7] 
            
            # Maximum Loss is max(WH, HH) - WO.
            flash[17] = (max(flash[8], flash[12]) - flash[7])/flash[7] 
            
    # Tally Total Flashes.
    df['Active'] = df['Bulls'] + df['Bears']
    df['Closed'] = df['BullsT'] + df['BearsT']
    
    # Format lists to pandas DataFrames.
    open_flashes_df = convert_list_to_dataframe(open_flashes)
    closed_flashes_df = convert_list_to_dataframe(closed_flashes)
    stunted_flashes_df = convert_list_to_dataframe(stunted_flashes)
    
    # Return DataFrames
    return df, open_flashes_df, closed_flashes_df, stunted_flashes_df

# %% Testing.
analysed_df, open_df, closed_df, stunted_df = adx_flash_analysis(csv1,
                                                                 14,
                                                                 12,
                                                                 7, 
                                                                 20, 
                                                                 2)

# %% Export DataFrames as .csv.

closed_df.to_csv('Y1closed.csv')









    