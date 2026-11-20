//! Serde helpers for optional const-generic joint arrays on [`CuCycleSample`](crate::CuCycleSample).

macro_rules! impl_option_array_serde {
    ($mod_name:ident, $elem_ty:ty, $inner_mod:ident, $label:literal) => {
        #[doc = concat!("Serde `with` module for `Option<[", $label, "; N]>` fields.")]
        pub(crate) mod $mod_name {
            use core::fmt;
            use core::marker::PhantomData;
            use serde::Deserialize;
            use serde::Deserializer;
            use serde::Serialize;
            use serde::Serializer;
            use serde::de::Error as DeError;
            use serde::de::Visitor;

            struct ArrayRef<'a, const N: usize>(&'a [$elem_ty; N]);

            impl<const N: usize> Serialize for ArrayRef<'_, N> {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    repose_kinematics::serde_arrays::$inner_mod::serialize(self.0, serializer)
                }
            }

            struct ArrayVal<const N: usize>([$elem_ty; N]);

            impl<'de, const N: usize> Deserialize<'de> for ArrayVal<N> {
                fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    repose_kinematics::serde_arrays::$inner_mod::deserialize(deserializer).map(Self)
                }
            }

            pub fn serialize<S, const N: usize>(
                value: &Option<[$elem_ty; N]>,
                serializer: S,
            ) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                match value {
                    Some(arr) => serializer.serialize_some(&ArrayRef(arr)),
                    None => serializer.serialize_none(),
                }
            }

            pub fn deserialize<'de, D, const N: usize>(
                deserializer: D,
            ) -> Result<Option<[$elem_ty; N]>, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct OptVisitor<const N: usize>(PhantomData<[$elem_ty; N]>);

                impl<'de, const N: usize> Visitor<'de> for OptVisitor<N> {
                    type Value = Option<[$elem_ty; N]>;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(formatter, "an optional array of {N} {} values", $label)
                    }

                    fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
                        Ok(None)
                    }

                    fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
                        Ok(None)
                    }

                    fn visit_some<D2: Deserializer<'de>>(
                        self,
                        deserializer: D2,
                    ) -> Result<Self::Value, D2::Error> {
                        ArrayVal::<N>::deserialize(deserializer).map(|a| Some(a.0))
                    }
                }

                deserializer.deserialize_option(OptVisitor::<N>(PhantomData))
            }
        }
    };
}

impl_option_array_serde!(
    option_position_array,
    repose_kinematics::CuJointPosition,
    position_array,
    "CuJointPosition"
);

impl_option_array_serde!(
    option_velocity_array,
    repose_kinematics::CuJointVelocity,
    velocity_array,
    "CuJointVelocity"
);
