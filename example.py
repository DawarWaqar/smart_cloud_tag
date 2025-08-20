#!/usr/bin/env python3


from dotenv import load_dotenv
from smart_cloud_tag import SmartCloudTagger


def main():
    """Main example function."""

    # Load environment variables from .env file
    load_dotenv()

    print("🚀 Smart Cloud Tag Example")
    print("=" * 50)

    try:

        tags = {
            "protected_health_information": ["true", "false"],
            "document_type": [
                "support_email",
                "claim_or_prior_auth",
                "lab_summary",
                "chat_transcript",
            ],
            "department": ["primary_care", "insurance", "labs", "billing"],
            "contains_billing": ["true", "false"],
        }

        custom_prompt_template = """
            You are a healthcare data classifier. Analyze this file and assign tags.
            
            FILENAME: {filename}
            CONTENT: {content}
            TAGS TO ASSIGN: {tags}
            
            Instructions: Focus on PHI detection and healthcare context.
            Return only the tag values separated by commas.
            """

        storage_uri_aws = "s3://smartcloudtagtestbucket"

        storage_uri_azure = "az://smartcloudtagtestbucket"

        storage_uri_gcp = "gs://smartcloudtagtest"

        # Initialize the tagger
        tagger = SmartCloudTagger(
            storage_uri=storage_uri_azure,  # target bucket location (supports aws/azure/gcp) (required)
            tags=tags,  # tag schema (required)
            storage_provider="azure",
            # llm_provider,  # openai/anthropic/gemini (default: openai) (optional)
            # llm_model, # (default: best performing model for the chosen llm provider) (optional)
            # custom_prompt_template=custom_prompt_template, # custom prompt template (optional)
            # max_bytes=10000, # max bytes to read from each file (optional)
        )

        # Apply tags
        tagger.apply_tags()

        print(f"📦 Storage: {tagger.get_storage_info()}")
        print(f"🤖 LLM: {tagger.get_llm_info()}")
        print(f"🏷️  Tags: {tagger.get_tags_info()}")

        # Preview tags without applying them
        print("🔍 Previewing tags...")
        preview_result = tagger.preview_tags()

        print(f"📊 Preview Results:")
        print(f"   Total objects: {preview_result.summary.get('total_objects', 0)}")
        print(f"   Processed: {preview_result.summary.get('processed', 0)}")
        print(f"   Skipped: {preview_result.summary.get('skipped', 0)}")

        for uri, result in list(preview_result.results.items()):
            print(f"📄 {uri}")
            if result.proposed:
                print(f"   Proposed tags: {result.proposed}")
            if result.skipped_reason:
                print(f"   Skipped: {result.skipped_reason}")
            print()

        response = input("❓ Do you want to apply these tags? (y/N): ").strip().lower()

        if response in ["y", "yes"]:
            print("✅ Applying tags...")
            apply_result = tagger.apply_tags()

            print(f"📊 Apply Results:")
            print(f"   Total objects: {apply_result.summary.get('total_objects', 0)}")
            print(f"   Applied: {apply_result.summary.get('applied', 0)}")
            print(f"   Skipped: {apply_result.summary.get('skipped', 0)}")

            print()
            print("🎉 Tags applied successfully!")

        else:
            print("⏭️  Skipping tag application")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
