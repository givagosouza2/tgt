import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Análise de Acelerômetro e Giroscópio",
    layout="wide"
)

st.title("Análise de Acelerômetro e Giroscópio")

st.markdown(
    """
    Aplicação para processamento de sinais inerciais:

    **Acelerômetro**
    - Detrend
    - Normalização em g quando necessário
    - Interpolação para 100 Hz
    - Cálculo da norma
    - Filtragem
    - Ajuste hiperbólico opcional

    **Giroscópio**
    - Detrend
    - Interpolação para 100 Hz
    - Cálculo da norma
    - Filtro passa-banda 1–4 Hz
    - Detecção e contagem de picos
    """
)


# ============================================================
# FUNÇÕES
# ============================================================

def michaelis_menten(x, DC, Vmax, n, Km):
    """
    Modelo hiperbólico/Hill.
    """
    return DC + Vmax * (x ** n) / (x ** n + Km ** n)


def butterworth_filter(data, cutoff, fs, order=4, btype="low"):
    """
    Filtro Butterworth genérico.
    cutoff:
        float para low/high
        [low, high] para bandpass
    """

    nyquist = 0.5 * fs

    if btype == "bandpass":
        normal_cutoff = [
            cutoff[0] / nyquist,
            cutoff[1] / nyquist
        ]
    else:
        normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype=btype,
        analog=False
    )

    return filtfilt(b, a, data)


def preprocess_sensor(
    df,
    fs=100,
    normalize_gravity=False
):
    """
    Pré-processamento comum aos sensores.

    Assume:
    coluna 0 = tempo em ms
    coluna 1 = eixo X
    coluna 2 = eixo Y
    coluna 3 = eixo Z

    Retorna:
    tempo em segundos
    X
    Y
    Z
    norma
    """

    time_raw = pd.to_numeric(
        df.iloc[:, 0],
        errors="coerce"
    ).to_numpy(dtype=float)

    x_raw = pd.to_numeric(
        df.iloc[:, 1],
        errors="coerce"
    ).to_numpy(dtype=float)

    y_raw = pd.to_numeric(
        df.iloc[:, 2],
        errors="coerce"
    ).to_numpy(dtype=float)

    z_raw = pd.to_numeric(
        df.iloc[:, 3],
        errors="coerce"
    ).to_numpy(dtype=float)

    # Remover NaN
    valid = (
        np.isfinite(time_raw)
        & np.isfinite(x_raw)
        & np.isfinite(y_raw)
        & np.isfinite(z_raw)
    )

    time_raw = time_raw[valid]
    x_raw = x_raw[valid]
    y_raw = y_raw[valid]
    z_raw = z_raw[valid]

    # Garantir ordenação temporal
    order = np.argsort(time_raw)

    time_raw = time_raw[order]
    x_raw = x_raw[order]
    y_raw = y_raw[order]
    z_raw = z_raw[order]

    # Remover timestamps duplicados
    time_unique, unique_indices = np.unique(
        time_raw,
        return_index=True
    )

    time_raw = time_unique
    x_raw = x_raw[unique_indices]
    y_raw = y_raw[unique_indices]
    z_raw = z_raw[unique_indices]

    # --------------------------------------------------------
    # Normalização do acelerômetro para g, quando necessário
    # --------------------------------------------------------

    if normalize_gravity:

        max_abs = np.max(
            np.abs(
                np.concatenate(
                    [x_raw, y_raw, z_raw]
                )
            )
        )

        if max_abs > 9:
            x_raw = x_raw / 9.81
            y_raw = y_raw / 9.81
            z_raw = z_raw / 9.81

    # --------------------------------------------------------
    # Detrend
    # --------------------------------------------------------

    x_detrend = signal.detrend(x_raw)
    y_detrend = signal.detrend(y_raw)
    z_detrend = signal.detrend(z_raw)

    # --------------------------------------------------------
    # Interpolação para frequência desejada
    # --------------------------------------------------------

    # O arquivo original utiliza tempo em milissegundos.
    dt_ms = 1000 / fs

    time_interp_ms = np.arange(
        time_raw[0],
        time_raw[-1],
        dt_ms
    )

    interp_x = scipy.interpolate.interp1d(
        time_raw,
        x_detrend,
        kind="linear"
    )

    interp_y = scipy.interpolate.interp1d(
        time_raw,
        y_detrend,
        kind="linear"
    )

    interp_z = scipy.interpolate.interp1d(
        time_raw,
        z_detrend,
        kind="linear"
    )

    x = interp_x(time_interp_ms)
    y = interp_y(time_interp_ms)
    z = interp_z(time_interp_ms)

    # Converter ms para segundos
    t = time_interp_ms / 1000

    # --------------------------------------------------------
    # Norma
    # --------------------------------------------------------

    norm = np.sqrt(
        x ** 2
        + y ** 2
        + z ** 2
    )

    return t, x, y, z, norm


# ============================================================
# PARÂMETROS GERAIS
# ============================================================

FS = 100


# ============================================================
# ABAS
# ============================================================

tab_acc, tab_gyro = st.tabs(
    [
        "Acelerômetro",
        "Giroscópio"
    ]
)


# ============================================================
# ACELERÔMETRO
# ============================================================

with tab_acc:

    st.header("Acelerômetro")

    uploaded_acc = st.file_uploader(
        "Carregue o arquivo do acelerômetro",
        type=["txt", "csv"],
        key="accelerometer"
    )

    # --------------------------------------------------------
    # Controles
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        acc_cutoff = st.number_input(
            "Frequência de corte do acelerômetro (Hz)",
            min_value=0.1,
            max_value=20.0,
            value=0.5,
            step=0.1
        )

    with col2:

        use_hyperbolic_model = st.checkbox(
            "Aplicar modelos hiperbólicos",
            value=True
        )

    if uploaded_acc is not None:

        try:

            # ------------------------------------------------
            # Leitura
            # ------------------------------------------------

            df_acc = pd.read_csv(
                uploaded_acc,
                sep=";"
            )

            if df_acc.shape[1] < 4:
                st.error(
                    "O arquivo deve possuir pelo menos quatro "
                    "colunas: tempo, X, Y e Z."
                )
                st.stop()

            # ------------------------------------------------
            # Pré-processamento
            # ------------------------------------------------

            (
                t_acc,
                acc_x,
                acc_y,
                acc_z,
                acc_norm
            ) = preprocess_sensor(
                df_acc,
                fs=FS,
                normalize_gravity=True
            )

            # ------------------------------------------------
            # Filtro do acelerômetro
            # ------------------------------------------------

            acc_x_filtered = butterworth_filter(
                acc_x,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_y_filtered = butterworth_filter(
                acc_y,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_z_filtered = butterworth_filter(
                acc_z,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            acc_norm_filtered = butterworth_filter(
                acc_norm,
                acc_cutoff,
                FS,
                order=2,
                btype="low"
            )

            # ------------------------------------------------
            # Gráfico básico
            # ------------------------------------------------

            st.subheader("Norma da aceleração")

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                t_acc,
                acc_norm_filtered,
                label="Norma filtrada"
            )

            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração (g)")
            ax.set_title(
                "Norma da aceleração"
            )

            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            plt.close(fig)

            # ====================================================
            # MODELO HIPERBÓLICO
            # ====================================================

            if use_hyperbolic_model:

                st.subheader(
                    "Modelos hiperbólicos"
                )

                lim1 = 100

                if len(acc_norm_filtered) <= (
                    2 * lim1 + 10
                ):
                    st.warning(
                        "O sinal é muito curto para a "
                        "aplicação do modelo."
                    )

                else:

                    try:

                        # ----------------------------------------
                        # Pico máximo
                        # ----------------------------------------

                        peak_index = np.argmax(
                            acc_norm_filtered
                        )

                        if (
                            peak_index <= lim1
                            or peak_index >=
                            len(t_acc) - lim1
                        ):
                            raise ValueError(
                                "O pico principal está muito "
                                "próximo do início ou do final "
                                "do registro."
                            )

                        # ========================================
                        # MODELO DO FINAL
                        # ========================================

                        x_data = (
                            t_acc[
                                peak_index:
                                len(t_acc) - lim1
                            ]
                            - t_acc[peak_index]
                        )

                        y_data = acc_norm_filtered[
                            peak_index:
                            len(t_acc) - lim1
                        ]

                        # Inverter para ajuste
                        y_data_reverse = y_data[::-1]

                        initial_guess = [
                            0.1,
                            0.3,
                            9,
                            3
                        ]

                        params_end, covariance = curve_fit(
                            michaelis_menten,
                            x_data,
                            y_data_reverse,
                            p0=initial_guess,
                            maxfev=50000
                        )

                        (
                            DC_fit,
                            Vmax_fit,
                            n_fit,
                            Km_fit
                        ) = params_end

                        y_fit = michaelis_menten(
                            x_data,
                            DC_fit,
                            Vmax_fit,
                            n_fit,
                            Km_fit
                        )

                        y_fit = y_fit[::-1]

                        x_data = (
                            x_data
                            + t_acc[peak_index]
                        )

                        # ========================================
                        # MODELO DO INÍCIO
                        # ========================================

                        x_data2 = (
                            t_acc[
                                lim1:
                                peak_index
                            ]
                            - t_acc[lim1]
                        )

                        y_data2 = acc_norm_filtered[
                            lim1:
                            peak_index
                        ]

                        params_start, covariance = curve_fit(
                            michaelis_menten,
                            x_data2,
                            y_data2,
                            p0=initial_guess,
                            maxfev=50000
                        )

                        (
                            DC_fit2,
                            Vmax_fit2,
                            n_fit2,
                            Km_fit2
                        ) = params_start

                        y_fit2 = michaelis_menten(
                            x_data2,
                            DC_fit2,
                            Vmax_fit2,
                            n_fit2,
                            Km_fit2
                        )

                        x_data2 = (
                            x_data2
                            + t_acc[lim1]
                        )

                        # ========================================
                        # LIMIARES 25%, 50% E 75%
                        # ========================================

                        max_start = np.max(y_fit2)
                        max_end = np.max(y_fit)

                        # Início
                        idx_start_25 = np.where(
                            y_fit2 >
                            max_start * 0.25
                        )[0][0]

                        idx_start_50 = np.where(
                            y_fit2 >
                            max_start * 0.50
                        )[0][0]

                        idx_start_75 = np.where(
                            y_fit2 >
                            max_start * 0.75
                        )[0][0]

                        # Final
                        idx_end_25 = np.where(
                            y_fit <
                            max_end * 0.25
                        )[0][0]

                        idx_end_50 = np.where(
                            y_fit <
                            max_end * 0.50
                        )[0][0]

                        idx_end_75 = np.where(
                            y_fit <
                            max_end * 0.75
                        )[0][0]

                        # ========================================
                        # GRÁFICO
                        # ========================================

                        fig, ax = plt.subplots(
                            figsize=(12, 5)
                        )

                        ax.plot(
                            t_acc,
                            acc_norm_filtered,
                            label="Norma",
                            linewidth=1.5
                        )

                        ax.plot(
                            x_data2,
                            y_fit2,
                            label="Modelo inicial",
                            linewidth=2
                        )

                        ax.plot(
                            x_data,
                            y_fit,
                            label="Modelo final",
                            linewidth=2
                        )

                        # Limiares
                        ax.axvline(
                            x_data2[
                                idx_start_25
                            ],
                            linestyle="--"
                        )

                        ax.axvline(
                            x_data2[
                                idx_start_75
                            ],
                            linestyle="--"
                        )

                        ax.axvline(
                            x_data[
                                idx_end_75
                            ],
                            linestyle="--"
                        )

                        ax.axvline(
                            x_data[
                                idx_end_25
                            ],
                            linestyle="--"
                        )

                        ax.set_xlabel("Tempo (s)")
                        ax.set_ylabel(
                            "Aceleração (g)"
                        )

                        ax.set_title(
                            "Ajustes hiperbólicos"
                        )

                        ax.legend()
                        ax.grid(alpha=0.3)

                        st.pyplot(fig)

                        plt.close(fig)

                        # ========================================
                        # CÁLCULOS
                        # ========================================

                        amplitude_acc = Vmax_fit2
                        amplitude_dec = Vmax_fit

                        tempo_acc = (
                            x_data2[idx_start_75]
                            - x_data2[idx_start_25]
                        )

                        tempo_dec = (
                            x_data[idx_end_25]
                            - x_data[idx_end_75]
                        )

                        tempo_constante = (
                            x_data[idx_end_75]
                            - x_data2[idx_start_75]
                        )

                        # Ganho da aceleração
                        elapsed_start_50 = (
                            x_data2[idx_start_50]
                            - x_data2[0]
                        )

                        if elapsed_start_50 > 0:

                            ganho_acc = (
                                y_fit2[idx_start_50]
                                / elapsed_start_50
                            )

                        else:
                            ganho_acc = np.nan

                        # Ganho da desaceleração
                        elapsed_end_50 = (
                            x_data[idx_end_50]
                            - t_acc[peak_index]
                        )

                        if elapsed_end_50 > 0:

                            ganho_dec = (
                                y_fit[idx_end_50]
                                / elapsed_end_50
                            )

                        else:
                            ganho_dec = np.nan

                        tempo_total = (
                            x_data[idx_end_25]
                            - x_data2[idx_start_25]
                        )

                        if tempo_total > 0:
                            velocidade = 4 / tempo_total
                        else:
                            velocidade = np.nan

                        # ========================================
                        # APRESENTAÇÃO DOS RESULTADOS
                        # ========================================

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Amplitude máxima de aceleração",
                            f"{amplitude_acc:.3f} g"
                        )

                        col2.metric(
                            "Amplitude máxima de desaceleração",
                            f"{amplitude_dec:.3f} g"
                        )

                        col3.metric(
                            "Tempo total",
                            f"{tempo_total:.2f} s"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Tempo de aceleração",
                            f"{tempo_acc:.2f} s"
                        )

                        col2.metric(
                            "Tempo de desaceleração",
                            f"{tempo_dec:.2f} s"
                        )

                        col3.metric(
                            "Tempo de velocidade constante",
                            f"{tempo_constante:.2f} s"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Ganho de aceleração",
                            f"{ganho_acc:.4f} g/s"
                        )

                        col2.metric(
                            "Ganho de desaceleração",
                            f"{ganho_dec:.4f} g/s"
                        )

                        if (
                            np.isfinite(ganho_acc)
                            and np.isfinite(ganho_dec)
                            and ganho_dec != 0
                        ):
                            razao = (
                                ganho_acc / ganho_dec
                            )
                        else:
                            razao = np.nan

                        col3.metric(
                            "Razão dos ganhos",
                            f"{razao:.4f}"
                        )

                        st.metric(
                            "Velocidade média de caminhada",
                            f"{velocidade:.2f} m/s"
                        )

                        # ----------------------------------------
                        # Parâmetros dos modelos
                        # ----------------------------------------

                        with st.expander(
                            "Parâmetros dos modelos"
                        ):

                            model_table = pd.DataFrame(
                                {
                                    "Parâmetro": [
                                        "DC",
                                        "Vmax",
                                        "n",
                                        "Km"
                                    ],
                                    "Aceleração": [
                                        DC_fit2,
                                        Vmax_fit2,
                                        n_fit2,
                                        Km_fit2
                                    ],
                                    "Desaceleração": [
                                        DC_fit,
                                        Vmax_fit,
                                        n_fit,
                                        Km_fit
                                    ]
                                }
                            )

                            st.dataframe(
                                model_table,
                                use_container_width=True
                            )

                    except Exception as e:

                        st.error(
                            "Não foi possível ajustar os "
                            "modelos hiperbólicos."
                        )

                        st.write(
                            f"Motivo: {e}"
                        )

            else:

                st.info(
                    "Os modelos hiperbólicos estão "
                    "desativados."
                )

        except Exception as e:

            st.error(
                f"Erro ao processar acelerômetro: {e}"
            )


# ============================================================
# GIROSCÓPIO
# ============================================================

with tab_gyro:

    st.header("Giroscópio")

    uploaded_gyro = st.file_uploader(
        "Carregue o arquivo do giroscópio",
        type=["txt", "csv"],
        key="gyroscope"
    )

    st.subheader(
        "Parâmetros da detecção de picos"
    )

    col1, col2 = st.columns(2)

    with col1:

        gyro_prominence = st.number_input(
            "Proeminência mínima",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.3f"
        )

    with col2:

        min_peak_distance_seconds = st.number_input(
            "Distância mínima entre picos (s)",
            min_value=0.05,
            max_value=5.0,
            value=0.30,
            step=0.05
        )

    # Frequências fixadas conforme solicitado
    gyro_lowcut = 1.0
    gyro_highcut = 4.0

    st.caption(
        "Filtro do giroscópio: "
        "Butterworth passa-banda 1–4 Hz."
    )

    if uploaded_gyro is not None:

        try:

            # ------------------------------------------------
            # Leitura
            # ------------------------------------------------

            df_gyro = pd.read_csv(
                uploaded_gyro,
                sep=";"
            )

            if df_gyro.shape[1] < 4:

                st.error(
                    "O arquivo deve possuir pelo menos quatro "
                    "colunas: tempo, X, Y e Z."
                )

                st.stop()

            # ------------------------------------------------
            # Pré-processamento
            # ------------------------------------------------

            (
                t_gyro,
                gyro_x,
                gyro_y,
                gyro_z,
                gyro_norm
            ) = preprocess_sensor(
                df_gyro,
                fs=FS,
                normalize_gravity=False
            )

            # ------------------------------------------------
            # Filtro passa-banda da norma
            # ------------------------------------------------

            gyro_norm_filtered = butterworth_filter(
                gyro_norm,
                [gyro_lowcut, gyro_highcut],
                FS,
                order=4,
                btype="bandpass"
            )

            # Também filtrar eixos, caso sejam úteis depois
            gyro_x_filtered = butterworth_filter(
                gyro_x,
                [gyro_lowcut, gyro_highcut],
                FS,
                order=4,
                btype="bandpass"
            )

            gyro_y_filtered = butterworth_filter(
                gyro_y,
                [gyro_lowcut, gyro_highcut],
                FS,
                order=4,
                btype="bandpass"
            )

            gyro_z_filtered = butterworth_filter(
                gyro_z,
                [gyro_lowcut, gyro_highcut],
                FS,
                order=4,
                btype="bandpass"
            )

            # ====================================================
            # DETECÇÃO DE PICOS
            # ====================================================

            min_peak_distance_samples = int(
                min_peak_distance_seconds
                * FS
            )

            peaks, properties = find_peaks(
                gyro_norm_filtered,
                prominence=gyro_prominence,
                distance=min_peak_distance_samples
            )

            number_of_peaks = len(peaks)

            # ====================================================
            # RESULTADO
            # ====================================================

            st.metric(
                "Número de picos detectados",
                number_of_peaks
            )

            # ------------------------------------------------
            # Frequência média dos picos
            # ------------------------------------------------

            if number_of_peaks > 1:

                peak_intervals = np.diff(
                    t_gyro[peaks]
                )

                mean_peak_interval = np.mean(
                    peak_intervals
                )

                peak_frequency = (
                    1 / mean_peak_interval
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Intervalo médio entre picos",
                    f"{mean_peak_interval:.3f} s"
                )

                col2.metric(
                    "Frequência média",
                    f"{peak_frequency:.2f} Hz"
                )

            # ====================================================
            # GRÁFICO DA NORMA DO GIROSCÓPIO
            # ====================================================

            st.subheader(
                "Norma do giroscópio e picos detectados"
            )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.plot(
                t_gyro,
                gyro_norm_filtered,
                label="Norma filtrada 1–4 Hz",
                linewidth=1.3
            )

            ax.scatter(
                t_gyro[peaks],
                gyro_norm_filtered[peaks],
                marker="o",
                s=45,
                label=f"Picos (n={number_of_peaks})"
            )

            ax.set_xlabel(
                "Tempo (s)"
            )

            ax.set_ylabel(
                "Velocidade angular"
            )

            ax.set_title(
                "Norma do giroscópio – filtro passa-banda 1–4 Hz"
            )

            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            plt.close(fig)

            # ====================================================
            # GRÁFICO DOS EIXOS
            # ====================================================

            with st.expander(
                "Visualizar eixos do giroscópio"
            ):

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                ax.plot(
                    t_gyro,
                    gyro_x_filtered,
                    label="X"
                )

                ax.plot(
                    t_gyro,
                    gyro_y_filtered,
                    label="Y"
                )

                ax.plot(
                    t_gyro,
                    gyro_z_filtered,
                    label="Z"
                )

                ax.set_xlabel(
                    "Tempo (s)"
                )

                ax.set_ylabel(
                    "Velocidade angular"
                )

                ax.set_title(
                    "Giroscópio – eixos filtrados 1–4 Hz"
                )

                ax.legend()
                ax.grid(alpha=0.3)

                st.pyplot(fig)

                plt.close(fig)

            # ====================================================
            # TABELA DOS PICOS
            # ====================================================

            if number_of_peaks > 0:

                peak_table = pd.DataFrame(
                    {
                        "Pico": np.arange(
                            1,
                            number_of_peaks + 1
                        ),
                        "Tempo (s)": t_gyro[peaks],
                        "Amplitude": gyro_norm_filtered[
                            peaks
                        ],
                        "Proeminência": properties[
                            "prominences"
                        ]
                    }
                )

                with st.expander(
                    "Tabela dos picos detectados"
                ):

                    st.dataframe(
                        peak_table,
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Erro ao processar giroscópio: {e}"
            )
